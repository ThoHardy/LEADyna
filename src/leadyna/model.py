
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
import numpy as np
from joblib import Parallel, delayed
from filterpy.kalman import UnscentedKalmanFilter as UKF, JulierSigmaPoints
import json

from .datasets import LatentSeries

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



def _propagate(fx, x, dt, u_col, vector_inputs=True):
    """Apply ``fx`` to a vector of states whose inputs are ``u_col``.

    With ``vector_inputs`` (all built-in models), ``fx`` is element-wise in both
    ``x`` and ``u`` and is called once. Otherwise ``fx`` only has to be element-wise
    in ``x``: it receives a scalar input, once when all trials share the same input
    at this step, else once per distinct input value.
    """
    if vector_inputs:
        return np.broadcast_to(np.asarray(fx(x, dt, u_col), dtype=float), x.shape)
    u0 = u_col[0]
    if np.all(u_col == u0):
        return np.broadcast_to(np.asarray(fx(x, dt, u0), dtype=float), x.shape)
    out = np.empty_like(x)
    for val in np.unique(u_col):
        mask = u_col == val
        out[mask] = fx(x[mask], dt, val)
    return out


def compute_ukf_loglikelihood_fast(
    state_series: dict[int, np.ndarray],
    input_series: dict[int, np.ndarray],
    dt: float,
    process_noise_scalar: float,
    measure_noise_scalar: float,
    transition_function_factory: callable,
    vector_inputs: bool = True,
) -> float:
    """Same filter as :func:`compute_ukf_loglikelihood`, vectorized over trials.

    The recursion stays sequential in time but advances all trials of a category
    together. In 1-D with Julier sigma points and kappa=0 the sigma set is
    {x, x+sqrt(P), x-sqrt(P)} with weights {0, 1/2, 1/2}, so a UKF step is two
    evaluations of the transition function; the update re-draws sigma points from
    the predicted (x, P), as the filterpy engine does, which for the identity
    observation gives a Kalman-form update. Agrees with the filterpy engine to
    ~1e-11 on the total log-likelihood, a few hundred times faster on
    realistic data.
    """
    Q = process_noise_scalar**2 * dt
    R = measure_noise_scalar**2
    log2pi = np.log(2 * np.pi)
    total = 0.0
    for signal_category in input_series.keys():
        y = np.asarray(state_series[signal_category], dtype=float)
        u = np.asarray(input_series[signal_category], dtype=float)
        fx = transition_function_factory(signal_category)
        n = y.shape[0]
        x = y[:, 0].copy()
        P = np.ones(n)
        ll = 0.0
        for t in range(1, y.shape[1]):
            s = np.sqrt(P)
            f = _propagate(fx, np.concatenate((x + s, x - s)), dt,
                           np.concatenate((u[:, t - 1], u[:, t - 1])), vector_inputs)
            f_plus, f_minus = f[:n], f[n:]
            x_pred = 0.5 * (f_plus + f_minus)
            P_pred = 0.5 * ((f_plus - x_pred) ** 2 + (f_minus - x_pred) ** 2) + Q
            S = P_pred + R
            innov = y[:, t] - x_pred
            ll += -0.5 * np.sum(log2pi + np.log(S) + innov**2 / S)
            K = P_pred / S
            x = x_pred + K * innov
            P = P_pred - K * K * S
        total += ll
    return float(total)


ENGINES = ("fast", "filterpy")


def _check_engine(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}.")
    return engine


@contextmanager
def use_engine(engine: str):
    """Temporarily set the default likelihood engine of every model.

    ``with use_engine("filterpy"): ...`` reproduces LEAD / SOUNDMODEL numerics
    bit-for-bit, including inside the ``clever_fit_*`` functions, which build
    their own models.
    """
    previous = BaseLEADModel.engine
    BaseLEADModel.engine = _check_engine(engine)
    try:
        yield
    finally:
        BaseLEADModel.engine = previous


# =============================================================================
# LEAD Abstract Base Class
# =============================================================================

class BaseLEADModel(ABC):
    """
    Abstract base class for Latent Evidence Accumulation Dynamics (LEAD) models.

    The core is modality-agnostic: it consumes pre-extracted latent time-series
    (a :class:`~leadyna.datasets.LatentSeries` or a legacy ``(state, input)``
    dict pair keyed by ``signal_category``) and knows nothing about EEG channels,
    decimation or acquisition metadata. Category-specific models are free in the
    number of categories they carry (see ``n_categories``), so the same core
    serves 2-category iEEG paradigms and 7-level SNR designs alike.
    """
    
    _param_names = ['tau', 'process_noise', 'measure_noise']

    # Likelihood engine: "fast" (numpy, vectorized over trials) or "filterpy" (the
    # LEAD engine, bit-for-bit reproduction of SOUNDMODEL). Set per call (engine=),
    # per model (model.engine = ...) or globally (leadyna.use_engine(...)).
    engine = "fast"
    # True when every _make_fx transition is element-wise in both x and u (all built-in
    # models). A custom model whose fx only accepts a scalar u sets this to False; the
    # fast engine then groups trials by input value (slower with trial-varying inputs).
    vector_inputs = True

    def __init__(self, tau: float, process_noise: float, measure_noise: float):
        self.tau = tau
        self.process_noise = process_noise
        self.measure_noise = measure_noise
        self.dt = 1  # Default dt, can be overridden simulation-side or here

    # --- Stratified-weight helper ----------------------------------------

    def _register_stratified(self, weights: dict, prefixes, n_categories: int) -> dict:
        """Set per-category weight attributes ``{prefix}{i}`` from ``weights``.

        Replaces the old hard-coded ``w0..w6`` signature: for each prefix (e.g.
        ``"w"``, ``"g"``) it creates ``n_categories`` attributes, defaulting each
        to 0, and validates that every supplied key matches an expected
        ``{prefix}{index}`` with ``index < n_categories``. Returns, per prefix,
        the ordered list of attribute names (used to build ``_param_names``).
        """
        allowed = {}
        for pfx in prefixes:
            for i in range(n_categories):
                allowed[f"{pfx}{i}"] = pfx
        for key in weights:
            if key not in allowed:
                raise TypeError(
                    f"{type(self).__name__} got an unexpected weight '{key}'. "
                    f"For n_categories={n_categories}, expected any of "
                    f"{sorted(allowed)}."
                )
        names = {pfx: [] for pfx in prefixes}
        for pfx in prefixes:
            for i in range(n_categories):
                setattr(self, f"{pfx}{i}", weights.get(f"{pfx}{i}", 0))
                names[pfx].append(f"{pfx}{i}")
        return names

    # --- Abstract Interface ---

    @abstractmethod
    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        pass

    @abstractmethod
    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        pass

    @abstractmethod
    def _make_fx(self, category: int):
        """Return a fast transition callable ``fx(x, dt, u)`` for ``category``."""
        ...

    def loglikelihood(self, state_series, input_series=None,
                      n_jobs: int = 8, batch_size: int = 20, engine: str | None = None) -> float:
        """Total UKF log-likelihood of the data under this model.

        Accepts either a :class:`~leadyna.datasets.LatentSeries` (whose ``dt``
        is adopted) or the legacy ``(state_series, input_series)`` dict pair.
        ``engine`` overrides ``self.engine`` ("fast" or "filterpy"); ``n_jobs``
        and ``batch_size`` only apply to the "filterpy" engine.
        """
        state_series, input_series = self._resolve_series(state_series, input_series)
        if _check_engine(engine or self.engine) == "fast":
            return compute_ukf_loglikelihood_fast(
                state_series, input_series,
                dt=self.dt,
                process_noise_scalar=self.process_noise,
                measure_noise_scalar=self.measure_noise,
                transition_function_factory=self._make_fx,
                vector_inputs=self.vector_inputs,
            )
        return compute_ukf_loglikelihood(
            state_series, input_series,
            dt=self.dt,
            process_noise_scalar=self.process_noise,
            measure_noise_scalar=self.measure_noise,
            transition_function_factory=self._make_fx,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

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
    
    def _resolve_series(self, state_series, input_series):
        """Accept a LatentSeries or a ``(state_series, input_series)`` dict pair.

        A LatentSeries routes through the contract: its ``dt`` is adopted and its
        arrays returned. The legacy dict-pair path is unchanged (``dt`` untouched),
        so existing code and numerical results are preserved exactly.
        """
        if isinstance(state_series, LatentSeries):
            if input_series is not None:
                raise TypeError(
                    "Pass a LatentSeries alone, or state_series and input_series "
                    "as separate dicts — not both."
                )
            if state_series.inputs is None:
                raise ValueError(
                    "This LatentSeries has no inputs yet: define your input hypothesis "
                    "I(t) and attach it with latent.with_inputs(inputs) before fitting."
                )
            self.dt = state_series.dt
            return state_series.states, state_series.inputs
        if input_series is None:
            raise TypeError(
                "input_series is required when state_series is a plain dict; "
                "pass a LatentSeries to route through the contract instead."
            )
        return state_series, input_series

    # --- Fitting Tools ---
    
    def fit(self, state_series, input_series=None,
            init_params=None, bounds=None, fixed_params=None,
            n_jobs=None, batch_size=None, feedback: bool = False, engine=None):
        """
        Maximize likelihood to fit the model parameters.
        
        Args:
            state_series: Dictionary mapping signal_category -> arrays of shape (n_trials, n_time_steps)
            input_series: Dictionary mapping signal_category -> arrays of shape (n_trials, n_time_steps)
            init_params: Initial parameter values (list in order of _param_names)
            bounds: Bounds for each parameter (list of tuples)
            fixed_params: List of parameter names to keep fixed during optimization
            feedback: Whether to print optimization feedback
            engine: Likelihood engine for this fit ("fast" or "filterpy"); defaults to self.engine
        """
        from scipy.optimize import minimize

        state_series, input_series = self._resolve_series(state_series, input_series)
        if init_params is None or bounds is None:
            raise ValueError("fit() requires init_params and bounds.")
        if fixed_params is None:
            fixed_params = []
        ll_kw = {}
        if n_jobs is not None:
            ll_kw['n_jobs'] = n_jobs
        if batch_size is not None:
            ll_kw['batch_size'] = batch_size
        if engine is not None:
            ll_kw['engine'] = _check_engine(engine)
        
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
            return -self.loglikelihood(state_series, input_series, **ll_kw)
        
        init_free = [init_params[i] for i in free_indices]
        bounds_free = [bounds[i] for i in free_indices]
        result = minimize(to_minimize, init_free, bounds=bounds_free, method="L-BFGS-B")
        
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

class LinearLEAD(BaseLEADModel):
    """
    Linear model with category-specific input weights.
    dx/dt = -x/tau + w_category * input

    ``n_categories`` sets how many category weights ``w0..w{n_categories-1}``
    exist (default 7, matching the original SNR design). ``baseline_category``
    (default 0) marks the resting/baseline category whose weight is fixed to 0
    by the clever-fit strategies.
    """

    def __init__(self, tau, process_noise, measure_noise,
                 *, n_categories=7, baseline_category=0, **weights):
        super().__init__(tau, process_noise, measure_noise)
        self.n_categories = n_categories
        self.baseline_category = baseline_category
        names = self._register_stratified(weights, ['w'], n_categories)
        self._param_names = BaseLEADModel._param_names + names['w']

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

    def loglikelihood_kalman(self, state_series, input_series=None) -> float:
        """Exact Kalman Filter implementation (Analytical Solution for verify)."""
        state_series, input_series = self._resolve_series(state_series, input_series)
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


class SigmoidFeedbackLEAD(BaseLEADModel):
    """
    Single-category nonlinear model (input weight + state-dependent sigmoid feedback).
    dx/dt = -x/tau + w * input + gain / (1 + exp(sharpness*(threshold - x)))
    """
    _param_names = BaseLEADModel._param_names + ['input_weight', 'gain', 'threshold', 'sharpness']

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


class StratifiedSigmoidFeedbackLEAD(BaseLEADModel):
    """
    Stratified version of SigmoidFeedbackLEAD (input weights vary by category).
    """

    def __init__(self, tau, process_noise, measure_noise, gain, threshold, sharpness,
                 *, n_categories=7, baseline_category=0, **weights):
        super().__init__(tau, process_noise, measure_noise)
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness
        self.n_categories = n_categories
        self.baseline_category = baseline_category
        names = self._register_stratified(weights, ['w'], n_categories)
        self._param_names = BaseLEADModel._param_names + names['w'] + ['gain', 'threshold', 'sharpness']

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


class AffineFeedbackLEAD(BaseLEADModel):
    """
    dx/dt = -x/tau + w*u + (a*x + b) * sigmoid(...)
    """
    _param_names = BaseLEADModel._param_names + ['input_weight', 'a', 'b', 'threshold', 'sharpness']

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


class StratifiedAffineFeedbackLEAD(BaseLEADModel):
    """
    Stratified AffineFeedbackLEAD.
    """

    def __init__(self, tau, process_noise, measure_noise, a, b, threshold, sharpness,
                 *, n_categories=7, baseline_category=0, **weights):
        super().__init__(tau, process_noise, measure_noise)
        self.a = a
        self.b = b
        self.threshold = threshold
        self.sharpness = sharpness
        self.n_categories = n_categories
        self.baseline_category = baseline_category
        names = self._register_stratified(weights, ['w'], n_categories)
        self._param_names = BaseLEADModel._param_names + names['w'] + ['a', 'b', 'threshold', 'sharpness']

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


class GainModulationLEAD(BaseLEADModel):
    """
    dx/dt = -x/tau + w*u + gain*u*sigmoid(...)
    """
    _param_names = BaseLEADModel._param_names + ['input_weight', 'gain', 'threshold', 'sharpness']

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


class StratifiedGainModulationLEAD(BaseLEADModel):
    """
    Stratified gain modulation.
    """

    def __init__(
        self, tau, process_noise, measure_noise,
        threshold, sharpness,
        *, n_categories=7, baseline_category=0, **weights
    ):
        super().__init__(tau, process_noise, measure_noise)
        self.threshold = threshold
        self.sharpness = sharpness
        self.n_categories = n_categories
        self.baseline_category = baseline_category
        names = self._register_stratified(weights, ['w', 'g'], n_categories)
        self._param_names = (
            BaseLEADModel._param_names + ['threshold', 'sharpness']
            + names['w'] + names['g']
        )

    def input_function(self, input_value, signal_category):
        w = getattr(self, f"w{signal_category}")
        return w * input_value

    def _gains(self):
        return np.array([getattr(self, f"g{i}") for i in range(self.n_categories)], dtype=float)

    @staticmethod
    def _gain_for(g_all, category, u):
        """g{int(category * u)}, element-wise over trials (u is 0/1: input off/on)."""
        idx = np.asarray(category * u).astype(int)
        if np.any(idx < 0) or np.any(idx >= len(g_all)):
            raise ValueError(
                f"Gain index int(category * u) out of range for category {category}: "
                "StratifiedGainModulationLEAD expects inputs u in [0, 1]."
            )
        return g_all[idx]

    def nonlinearity(self, state, input_value, signal_category):
        g = self._gain_for(self._gains(), signal_category, input_value)
        return g / (1 + np.exp(self.sharpness * (self.threshold - state)))

    def _make_fx(self, category):
        w_cat = getattr(self, f"w{category}") # for linear part
        g_all = self._gains()
        
        th, sh = self.threshold, self.sharpness
        tau, dt = self.tau, self.dt
        
        def fx(x, dt_val, u):
            g = self._gain_for(g_all, category, u)
            
            # Nonlinearity: g / (1 + exp...)
            nl = g / (1 + np.exp(sh * (th - x)))
            
            # Linear part: -x/tau + w*u
            dxdt = -x/tau + w_cat*u + nl
            
            return x + dxdt * dt
        return fx


# =============================================================================
# Backwards-compatible aliases (deprecated: prefer the *LEAD names)
# =============================================================================
# The old insider names still resolve so existing SOUNDMODEL notebooks and any
# saved code keep importing them, but they emit a DeprecationWarning pointing at
# the new public name. Resolves both ``model.StratifiedLinear`` and
# ``from leadyna.model import StratifiedLinear`` (PEP 562 module __getattr__).

_DEPRECATED_ALIASES = {
    'LEAD_abstract': 'BaseLEADModel',
    'StratifiedLinear': 'LinearLEAD',
    'NonLinear1': 'SigmoidFeedbackLEAD',
    'StratifiedNonLinear1': 'StratifiedSigmoidFeedbackLEAD',
    'NonLinear2': 'AffineFeedbackLEAD',
    'StratifiedNonLinear2': 'StratifiedAffineFeedbackLEAD',
    'GainModulation': 'GainModulationLEAD',
    'StratifiedGainModulation': 'StratifiedGainModulationLEAD',
}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"leadyna.model.{name} is a deprecated alias for "
            f"leadyna.model.{new_name}; update your code to the new name.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
