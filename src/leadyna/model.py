
import warnings
from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
from filterpy.kalman import UnscentedKalmanFilter as UKF, JulierSigmaPoints
import json

# =============================================================================
# UKF Engine (The Core Logic)
# =============================================================================

def compute_ukf_loglikelihood(
    state_series: dict[int, np.ndarray],
    input_series: dict[int, np.ndarray],
    dt: float,
    process_noise_scalar: float,
    measure_noise_scalar: float,
    transition_function_factory: callable,
    n_jobs: int = 8,
    batch_size: int = 20
) -> float:
    """
    Computes the total log-likelihood for a dataset using a Parallelized Unscented Kalman Filter.

    This engine handles:
    1. Parallel execution across batches of trials.
    2. Correct UKF initialization with JulierSigmaPoints (kappa=0.0).
    3. The critical 'Predict -> Regenerate Sigmas -> Update' loop to handle additive process noise correctly.

    Args:
        state_series: Dictionary mapping signal_category -> arrays of shape (n_trials, n_time_steps).
        input_series: Dictionary mapping signal_category -> arrays of shape (n_trials, n_time_steps).
        dt: Time step duration.
        process_noise_scalar: Standard deviation of process noise (sigma_q).
        measure_noise_scalar: Standard deviation of measurement noise (sigma_r).
        transition_function_factory: A function that takes `signal_category` and returns a callable `fx(x, dt, u)`.
                                     This creates the model-specific physics for each category.
        n_jobs: Number of parallel jobs.
        batch_size: Number of trials per batch.

    Returns:
        float: The total log-likelihood of the data given the model.
    """

    # Pre-compute matrices dependent on scalar noise parameters
    # Note: Q and R provided to UKF should be variances/covariances.
    # We follow the convention: Q = (process_noise)**2 * dt (discrete approximation)
    Q_matrix = np.eye(1) * (process_noise_scalar**2 * dt)
    R_matrix = np.eye(1) * (measure_noise_scalar**2)
    
    # Sigma Points Configuration
    # JulierSigmaPoints with kappa=0 is stable for 1D/small dimensions and matches KF behavior for linear systems.
    SIGMA_KAPPA = 0.0
    UKF_DIM_X = 1
    UKF_DIM_Z = 1

    # --- Worker Function for a Single Batch ---
    def process_batch(batch_states, batch_inputs, signal_category):
        
        # Instantiate the model dynamics for this specific category
        model_fx = transition_function_factory(signal_category)
        
        batch_ll = 0.0
        
        # Iterate over each trial in the batch
        for states, inputs in zip(batch_states, batch_inputs):
            
            # Initialize UKF for this trial
            sigmas = JulierSigmaPoints(n=UKF_DIM_X, kappa=SIGMA_KAPPA)
            
            # Define wrapper for UKF.fx to handle 'u' input
            def fx_wrapper(x, dt_val, u_val=0.0):
                # Ensure the model returns a 1D array to satisfy filterpy requirements
                return np.atleast_1d(model_fx(x, dt_val, u_val))

            def hx_wrapper(x):
                return np.atleast_1d(x)

            ukf = UKF(
                dim_x=UKF_DIM_X, 
                dim_z=UKF_DIM_Z, 
                fx=lambda x, dt_l, u=inputs[0]: fx_wrapper(x, dt_l, u),
                hx=hx_wrapper, 
                dt=dt, 
                points=sigmas
            )

            # Set Initial State and Covariances
            ukf.x = np.atleast_1d(states[0])   # Initial state from observation
            ukf.P = np.eye(UKF_DIM_X)          # Initial uncertainty (identity)
            ukf.Q = Q_matrix
            ukf.R = R_matrix

            trial_ll = 0.0

            # Time Step Loop
            # We start from t=1 because t=0 is the initial state
            for t in range(1, len(states)):
                input_val = inputs[t - 1]
                obs_val = states[t]

                # Update the transition function with the current input
                ukf.fx = lambda x, dt_l, u=input_val: fx_wrapper(x, dt_l, u)
                
                # 1. Predict
                ukf.predict()
                
                # 2. Fix: Regenerate Sigma Points
                # FilterPy reuses sigmas from predict (which don't include Q). 
                # We must regenerate them from the predicted X and P (which DO include Q).
                ukf.sigmas_f = ukf.points_fn.sigma_points(ukf.x, ukf.P)

                # 3. Update
                ukf.update(obs_val)
                
                # Accumulate Log-Likelihood
                trial_ll += ukf.log_likelihood

            batch_ll += trial_ll

        return batch_ll

    # --- Prepare Batches ---
    tasks = []
    for signal_category in input_series.keys():
        states_cat = state_series[signal_category]
        inputs_cat = input_series[signal_category]
        n_trials = states_cat.shape[0]
        
        for i in range(0, n_trials, batch_size):
            b_states = states_cat[i:i + batch_size]
            b_inputs = inputs_cat[i:i + batch_size]
            tasks.append((b_states, b_inputs, signal_category))

    # --- Run Parallel Execution ---
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_batch)(bs, bi, cat) for bs, bi, cat in tasks
    )

    return float(np.sum(results))


# =============================================================================
# LEAD Abstract Base Class (formerly UPP)
# =============================================================================

class LEAD_abstract(ABC):
    """
    Abstract base class for Latent Evidence Accumulation Dynamics (LEAD) models.
    """
    
    _param_names = ['tau', 'process_noise', 'measure_noise']

    def __init__(self, tau: float, process_noise: float, measure_noise: float):
        self.tau = tau
        self.process_noise = process_noise
        self.measure_noise = measure_noise
        self.dt = 1  # Default dt, can be overridden simulation-side or here

    # --- Abstract Interface ---

    @abstractmethod
    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        pass

    @abstractmethod
    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        pass

    @abstractmethod
    def loglikelihood(self, state_series: dict, input_series: dict) -> float:
        pass

    # --- Shared Physics "Core" ---

    def core(self, state, input_value, signal_category):
        """
        Continuous Differential Equation: dx/dt = -x/tau + f(input) + g(x, input)
        """
        dxdt = (
            -state / self.tau 
            + self.input_function(input_value, signal_category) 
            + self.nonlinearity(state, input_value, signal_category)
        )
        # Euler approximation for discrete step: x_next = x + dx/dt * dt
        # Note: This returns the NEXT state directly.
        return state + dxdt * self.dt

    # --- Simulation Tools ---
    def measure_simulations(self, input_series: dict, initial_states: dict = None):
        """Generates simulated data for the model."""
        simulations = {}
        for cat, inputs in input_series.items():
            
            n_sims, n_steps = inputs.shape
            states = np.zeros((n_sims, n_steps)) # Assuming x0 = 0
            if initial_states != None:
                states[:, 0] = initial_states[cat]
            
            # Pre-calculate process-noise scale
            noise_scale = self.process_noise * np.sqrt(self.dt)
            
            for t in range(n_steps - 1):
                # Deterministic step
                x_next_det = self.core(states[:, t], inputs[:, t], cat)
                # Add process noise
                states[:, t+1] = x_next_det + np.random.normal(0, noise_scale, size=n_sims)
            
            simulations[cat] = states + np.random.normal(0, self.measure_noise, size=states.shape)
        return simulations
    
    # --- Parameter Management ---

    def get_params(self):
        """Return dict of parameters for this model."""
        return {name: getattr(self, name) for name in self._param_names}
    
    def get_parameters(self):
        """Alias for get_params"""
        return self.get_params()

    def set_params(self, params: dict):
        """Update parameters from dict."""
        for name, value in params.items():
            if name in self._param_names:
                setattr(self, name, value)
    
    def set_parameters(self, params: dict):
        """Alias for set_params"""
        self.set_params(params)
    
    def set_params_from_list(self, params_list: list):
        """Update parameters from list, order follows _param_names."""
        for name, val in zip(self._param_names, params_list):
            setattr(self, name, val)

    def save_params(self, save_path: str) -> None:
        """Save parameters in a file given by the user."""
        with open(save_path, "w") as f:
            json.dump(self.get_params(), f)

    def load_params(self, save_path: str) -> None:
        """Load parameters from a file given by the user."""
        with open(save_path, "r") as f:
            params = json.load(f)
        self.set_params(params)
    
    # --- Fitting Tools ---
    
    def fit(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray], 
            init_params: list, bounds: list, fixed_params: list = [], 
            feedback: bool = False):
        """
        Maximize likelihood to fit the model parameters.
        
        Args:
            state_series: Dictionary mapping signal_category -> arrays of shape (n_trials, n_time_steps)
            input_series: Dictionary mapping signal_category -> arrays of shape (n_trials, n_time_steps)
            init_params: Initial parameter values (list in order of _param_names)
            bounds: Bounds for each parameter (list of tuples)
            fixed_params: List of parameter names to keep fixed during optimization
            feedback: Whether to print optimization feedback
        """
        from scipy.optimize import minimize
        
        # Identify which parameters to optimize
        free_indices = [i for i, name in enumerate(self._param_names) if name not in fixed_params]
        free_names = [self._param_names[i] for i in free_indices]
        if feedback:
            print(f'Fit model with free parameters {free_names}.')
        
        def to_minimize(p_free):
            full_params = init_params.copy()
            for i, idx in enumerate(free_indices):
                full_params[idx] = p_free[i]
            self.set_params_from_list(full_params)
            return -self.loglikelihood(state_series, input_series)
        
        init_free = [init_params[i] for i in free_indices]
        bounds_free = [bounds[i] for i in free_indices]
        result = minimize(to_minimize, init_free, bounds=bounds_free)
        
        # Apply the fitted parameters
        full_params = init_params.copy()
        for i, idx in enumerate(free_indices):
            full_params[idx] = result.x[i]
        self.set_params_from_list(full_params)
        
        if feedback:
            print(f'Optimisation success: {result.success}. \nFinal log-likelihood evaluation: {result.fun}.')


# =============================================================================
# Concrete Implementations
# =============================================================================

class StratifiedLinear(LEAD_abstract):
    """
    Standard Linear Model with category-specific weights.
    dx/dt = -x/tau + w_category * input
    """
    
    _param_names = LEAD_abstract._param_names + [f'w{i}' for i in range(7)]

    def __init__(self, tau, process_noise, measure_noise, 
                 w0=0, w1=0, w2=0, w3=0, w4=0, w5=0, w6=0):
        super().__init__(tau, process_noise, measure_noise)
        self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = w0, w1, w2, w3, w4, w5, w6

    def input_function(self, input_value, signal_category):
        w = getattr(self, f"w{signal_category}")
        return w * input_value

    def nonlinearity(self, state, input_value, signal_category):
        return 0.0

    def _make_fx(self, category):
        # Pre-calculate constants for efficiency
        w = getattr(self, f"w{category}")
        # Physics: x_next = x + (-x/tau + w*u)*dt
        #                 = (1 - dt/tau)*x + (w*dt)*u
        decay = 1.0 - self.dt / self.tau
        input_gain = w * self.dt
        
        # Return function with signature (x, dt, u)
        return lambda x, dt_val, u: decay * x + input_gain * u

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )

    def loglikelihood_kalman(self, state_series: dict, input_series: dict) -> float:
        """Exact Kalman Filter implementation (Analytical Solution for verify)."""
        total_ll = 0.0
        A = 1 - self.dt / self.tau
        Q = self.process_noise**2 * self.dt
        R = self.measure_noise**2
        
        for cat in input_series.keys():
            w = getattr(self, f"w{cat}")
            B = w * self.dt
            
            states_cat = state_series[cat]
            inputs_cat = input_series[cat]
            
            # Simple scalar Kalman Filter
            for states, inputs in zip(states_cat, inputs_cat):
                x, P = states[0], 1.0
                
                for t in range(1, len(states)):
                    # Predict
                    x = A * x + B * inputs[t-1]
                    P = A**2 * P + Q
                    
                    # Update
                    y = states[t] - x
                    S = P + R
                    K = P / S
                    x = x + K * y
                    P = (1 - K) * P
                    
                    total_ll += -0.5 * (np.log(2 * np.pi * S) + y**2 / S)
        return total_ll


class NonLinear1(LEAD_abstract):
    """
    Single-category Nonlinear Model (Input Weight + Sigmoid Feedback).
    dx/dt = -x/tau + w * input + gain / (1 + exp(slope*(threshold - x)))
    """
    _param_names = LEAD_abstract._param_names + ['input_weight', 'gain', 'threshold', 'sharpness']

    def __init__(self, tau, process_noise, measure_noise, input_weight, gain, threshold, sharpness):
        super().__init__(tau, process_noise, measure_noise)
        self.input_weight = input_weight
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness

    def input_function(self, input_value, signal_category):
        return self.input_weight * input_value

    def nonlinearity(self, state, input_value, signal_category):
        return self.gain / (1 + np.exp(self.sharpness * (self.threshold - state)))

    def _make_fx(self, category):
        # Closure over parameters
        iw, g, th, sh = self.input_weight, self.gain, self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            # Physics: dx/dt = -x/tau + iw*u + sigmoid
            sigmoid = g / (1 + np.exp(sh * (th - x)))
            dxdt = -x/tau + iw*u + sigmoid
            return x + dxdt * dt
        return fx

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )


class StratifiedNonLinear1(LEAD_abstract):
    """
    Stratified version of NonLinear1 (Input weights vary by category).
    """
    _param_names = LEAD_abstract._param_names + [f'w{i}' for i in range(7)] + ['gain', 'threshold', 'sharpness'] # I forgot sharpness but this may break compatibility when loading parameters, so IMPORTANT TO DO LATER (resolved but confirm)

    def __init__(self, tau, process_noise, measure_noise, gain, threshold, sharpness,
                 w0=0, w1=0, w2=0, w3=0, w4=0, w5=0, w6=0):
        super().__init__(tau, process_noise, measure_noise)
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness
        self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = w0, w1, w2, w3, w4, w5, w6

    def input_function(self, input_value, signal_category):
        w = getattr(self, f"w{signal_category}")
        return w * input_value

    def nonlinearity(self, state, input_value, signal_category):
        return self.gain / (1 + np.exp(self.sharpness * (self.threshold - state)))

    def _make_fx(self, category):
        w = getattr(self, f"w{category}")
        g, th, sh = self.gain, self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            sigmoid = g / (1 + np.exp(sh * (th - x)))
            dxdt = -x/tau + w*u + sigmoid
            return x + dxdt * dt
        return fx

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )


class NonLinear2(LEAD_abstract):
    """
    dx/dt = -x + w*u + (a*x + b) * sigmoid(...)
    """
    _param_names = LEAD_abstract._param_names + ['input_weight', 'a', 'b', 'threshold', 'sharpness']

    def __init__(self, tau, process_noise, measure_noise, input_weight, a, b, threshold, sharpness):
        super().__init__(tau, process_noise, measure_noise)
        self.input_weight = input_weight
        self.a = a
        self.b = b
        self.threshold = threshold
        self.sharpness = sharpness

    def input_function(self, input_value, signal_category):
        return self.input_weight * input_value

    def nonlinearity(self, state, input_value, signal_category):
        sigmoid = 1 / (1 + np.exp(self.sharpness * (self.threshold - state)))
        return (self.a * state + self.b) * sigmoid

    def _make_fx(self, category):
        iw, a, b, th, sh = self.input_weight, self.a, self.b, self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            sigmoid = 1 / (1 + np.exp(sh * (th - x)))
            nl = (a*x + b) * sigmoid
            dxdt = -x/tau + iw*u + nl
            return x + dxdt * dt
        return fx

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )


class StratifiedNonLinear2(LEAD_abstract):
    """
    Stratified NonLinear2.
    """
    _param_names = LEAD_abstract._param_names + [f'w{i}' for i in range(7)] + ['a', 'b', 'threshold', 'sharpness']

    def __init__(self, tau, process_noise, measure_noise, a, b, threshold, sharpness,
                 w0=0, w1=0, w2=0, w3=0, w4=0, w5=0, w6=0):
        super().__init__(tau, process_noise, measure_noise)
        self.a = a
        self.b = b
        self.threshold = threshold
        self.sharpness = sharpness
        self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = w0, w1, w2, w3, w4, w5, w6

    def input_function(self, input_value, signal_category):
        w = getattr(self, f"w{signal_category}")
        return w * input_value

    def nonlinearity(self, state, input_value, signal_category):
        sigmoid = 1 / (1 + np.exp(self.sharpness * (self.threshold - state)))
        return (self.a * state + self.b) * sigmoid

    def _make_fx(self, category):
        w = getattr(self, f"w{category}")
        a, b, th, sh = self.a, self.b, self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            sigmoid = 1 / (1 + np.exp(sh * (th - x)))
            nl = (a*x + b) * sigmoid
            dxdt = -x/tau + w*u + nl
            return x + dxdt * dt
        return fx

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )


class GainModulation(LEAD_abstract):
    """
    dx/dt = -x + w*u + gain*u*sigmoid(...)
    """
    _param_names = LEAD_abstract._param_names + ['input_weight', 'gain', 'threshold', 'sharpness']

    def __init__(self, tau, process_noise, measure_noise, input_weight, gain, threshold, sharpness):
        super().__init__(tau, process_noise, measure_noise)
        self.input_weight = input_weight
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness

    def input_function(self, input_value, signal_category):
        return self.input_weight * input_value

    def nonlinearity(self, state, input_value, signal_category):
        sigmoid = 1 / (1 + np.exp(self.sharpness * (self.threshold - state)))
        return self.gain * input_value * sigmoid

    def _make_fx(self, category):
        iw, g, th, sh = self.input_weight, self.gain, self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            sigmoid = 1 / (1 + np.exp(sh * (th - x)))
            nl = g * u * sigmoid
            dxdt = -x/tau + iw*u + nl
            return x + dxdt * dt
        return fx

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )


class StratifiedGainModulation(LEAD_abstract):
    """
    Stratified Gain Modulation.
    """
    _param_names = LEAD_abstract._param_names + ['threshold', 'sharpness'] + [f'w{i}' for i in range(7)] + [f'g{i}' for i in range(7)]

    def __init__(
        self, tau, process_noise, measure_noise,
        threshold, sharpness,
        w0=0, w1=0, w2=0, w3=0, w4=0, w5=0, w6=0,
        g0=0, g1=0, g2=0, g3=0, g4=0, g5=0, g6=0
    ):
        super().__init__(tau, process_noise, measure_noise)
        self.threshold = threshold
        self.sharpness = sharpness
        self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = w0, w1, w2, w3, w4, w5, w6
        self.g0, self.g1, self.g2, self.g3, self.g4, self.g5, self.g6 = g0, g1, g2, g3, g4, g5, g6

    def input_function(self, input_value, signal_category):
        w = getattr(self, f"w{signal_category}")
        return w * input_value

    def nonlinearity(self, state, input_value, signal_category):
        # Original logic: g depends on signal_category * input_on?
        # "input_on is supposed to always be 0 or 1"
        try:
            inp_scalar = input_value[0] if isinstance(input_value, np.ndarray) else input_value
        except:
            inp_scalar = input_value
        
        idx = int(signal_category * inp_scalar)
        g = getattr(self, f"g{idx}")
        return g / (1 + np.exp(self.sharpness * (self.threshold - state)))

    def _make_fx(self, category):
        w_cat = getattr(self, f"w{category}") # for linear part
        
        th, sh = self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            u_scalar = u[0] if isinstance(u, np.ndarray) else u
            idx = int(category * u_scalar)
            g = getattr(self, f"g{idx}")
            
            # Nonlinearity: g / (1 + exp...)
            nl = g / (1 + np.exp(sh * (th - x)))
            
            # Linear part: -x/tau + w*u
            dxdt = -x/tau + w_cat*u + nl
            
            return x + dxdt * dt
        return fx

    def loglikelihood(self, state_series, input_series):
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx
        )
