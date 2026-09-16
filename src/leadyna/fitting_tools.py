import numpy as np
from . import model as model


def clever_fit_null(state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedLinear:
    """
    Clever fitting strategy for StratifiedLinear model.
    
    Strategy:
    Fit tau, process_noise, measure_noise on all categories while fixing w{cat}=0 for each of them.
    """

    n_categories = len(list(state_train.keys()))

    # Isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    state_train_constant_stim[0] = state_train[0]
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim[0] = input_train[0]

    # Fit the OU process with the hypothesis of resting-state dynamics at each category (w{cat}=0)
    null = model.StratifiedLinear(tau=10, process_noise=0.1, measure_noise=0.1, w0=0)
    null.fit(
        state_series=state_train_constant_stim,
        input_series=input_train_constant_stim,
        init_params=[10, 0.1, 0.1] + [0]*7,  # tau, process_noise, measure_noise, w0...w6
        bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7,
        fixed_params=[f'w{i}' for i in range(7)])  # Fix all weights to 0 
    
    return null



def clever_fit_linear(state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedLinear:
    """
    Clever fitting strategy for StratifiedLinear model.
    
    Strategy:
    1. Fit tau, process_noise, measure_noise on category 0 (resting state, w0=0)
    2. For each category 1-6, fit the corresponding w_i on constant stimulation segments
    """

    n_categories = len(list(state_train.keys()))

    # Fit the OU process on resting-state dynamics (category 0)
    # We use a simple Linear model temporarily to get tau, process_noise, measure_noise
    linear_resting_state = model.StratifiedLinear(tau=10, process_noise=0.1, measure_noise=0.1, w0=0)
    linear_resting_state.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1] + [0]*7,  # tau, process_noise, measure_noise, w0...w6
        bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7,
        fixed_params=[f'w{i}' for i in range(7)])  # Fix all weights during this step
    
    # Isolate segments where stimulation is supposed constant (75-100 for categories 1+)
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}

    # Fit linear weights for each category
    linear = model.StratifiedLinear(
        tau=linear_resting_state.tau, 
        process_noise=linear_resting_state.process_noise, 
        measure_noise=linear_resting_state.measure_noise
    )
    
    for cat in range(1, n_categories):
        ws, lls = [], []
        for w_init in [0, 0.1]:
            # Create a temporary model for this category
            linear_one_cat = model.StratifiedLinear(
                tau=linear.tau, 
                process_noise=linear.process_noise, 
                measure_noise=linear.measure_noise
            )
            setattr(linear_one_cat, f'w{cat}', w_init)
            
            # Fit only the weight for this category
            init_params = [linear.tau, linear.process_noise, linear.measure_noise] + [0]*7
            init_params[3 + cat] = w_init
            
            linear_one_cat.fit(
                state_series={cat: state_train_constant_stim[cat]},
                input_series={cat: input_train_constant_stim[cat]},
                init_params=init_params,
                bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7,
                fixed_params=['tau', 'measure_noise', 'process_noise'] + [f'w{i}' for i in range(7) if i != cat])
            
            ws.append(getattr(linear_one_cat, f'w{cat}'))
            lls.append(linear_one_cat.loglikelihood({cat: state_train_constant_stim[cat]}, {cat: input_train_constant_stim[cat]}))
        
        linear.set_params({f'w{cat}': ws[np.argmax(lls)]})
    
    return linear


def clever_fit_gainmodul(linear_prefitted: model.StratifiedLinear, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedGainModulation:
    """
    Clever fitting strategy for StratifiedGainModulation model.
    
    Strategy:
    1. Initialize from prefitted linear model
    2. Grid search over threshold values
    3. For each threshold, alternate between:
       - Fitting (w, g) pairs for each category
       - Re-adjusting the threshold and tau
    """

    n_categories = len(list(state_train.keys()))

    # Fit the OU process on resting-state dynamics
    linear = model.StratifiedLinear(tau=10, process_noise=0.1, measure_noise=0.1)
    linear.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1] + [0]*7,
        bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7,
        fixed_params=[f'w{i}' for i in range(7)])

    # Isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}

    # Fit gainmodul (different initializations for the threshold)
    whole_GMmodels, ll_GMmodels = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(2):
            init_gain_multiplier = init_gain_index/2       # make gain_init start from 0 and w5_linear/2
            w_multiplier = 1 - init_gain_index/2           # make w5_init start from w5_linear and w5_linear/2
            
            gainmodul = model.StratifiedGainModulation(
                tau=linear.tau, 
                process_noise=linear.process_noise, 
                measure_noise=linear.measure_noise, 
                threshold=th, 
                sharpness=5,
                **{f"w{cat}": getattr(linear_prefitted, f"w{cat}")*w_multiplier for cat in range(1, n_categories)}, 
                **{f"g{cat}": getattr(linear_prefitted, f"w{cat}")*init_gain_multiplier for cat in range(1, n_categories)}
            )
            
            for _ in range(n_loops):
                # Focus on the (w, g) couples for all categories except 0
                for cat in range(1, n_categories):
                    gainmodul_one_cat = model.NonLinear1(
                        tau=gainmodul.tau, 
                        process_noise=gainmodul.process_noise, 
                        measure_noise=gainmodul.measure_noise,
                        input_weight=getattr(gainmodul, f'w{cat}'),
                        gain=getattr(gainmodul, f'g{cat}'),
                        threshold=gainmodul.threshold, 
                        sharpness=5
                    )
                    
                    gainmodul_one_cat.fit(
                        state_series={0: state_train_constant_stim[cat]},
                        input_series={0: input_train_constant_stim[cat]},
                        init_params=[getattr(gainmodul_one_cat, pname) for pname in gainmodul_one_cat._param_names],
                        bounds=[(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2), (0, 10)],
                        fixed_params=['tau', 'process_noise', 'measure_noise', 'threshold', 'sharpness'],
                        feedback=False
                    )
                    gainmodul.set_params({f'w{cat}': gainmodul_one_cat.input_weight, f'g{cat}': gainmodul_one_cat.gain})
                
                # Focus on the threshold and NOT tau
                gainmodul.fit(
                    state_series=state_train_constant_stim,
                    input_series=input_train_constant_stim,
                    init_params=[getattr(gainmodul, pname) for pname in gainmodul._param_names],
                    bounds=[(1,25), (0.01, 1), (0.01, 1), (0, 2), (0, 10)] + 14*[(0, 1)],
                    fixed_params=['tau', 'process_noise', 'measure_noise', 'sharpness'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(7)],
                    feedback=False
                )
            
            whole_GMmodels.append(gainmodul)
            ll_GMmodels.append(gainmodul.loglikelihood(state_train_constant_stim, input_train_constant_stim))
    
    gainmodul_fitted = whole_GMmodels[np.argmax(ll_GMmodels)]
    
    return gainmodul_fitted


def clever_fit_nonlinear1(linear_prefitted: model.StratifiedLinear, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedNonLinear1:
    """
    Clever fitting strategy for StratifiedNonLinear1 model.
    
    Strategy:
    1. Initialize from prefitted linear model
    2. Grid search over threshold values
    3. For each threshold, alternate between:
       - Fitting gain and threshold
       - Fitting input weights for each category
    """

    n_categories = len(list(state_train.keys()))

    # Fit the OU process on resting-state dynamics
    linear = model.StratifiedLinear(tau=10, process_noise=0.1, measure_noise=0.1)
    linear.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1] + [0]*7,
        bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7,
        fixed_params=[f'w{i}' for i in range(7)])

    # Isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    state_train_constant_stim[0] = state_train[0]
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim[0] = input_train[0]

    # Fit nonlinear (different initializations for the threshold)
    all_models, ll_models = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(2):
            init_gain_multiplier = init_gain_index/2       # make gain_init range from 0 to w5_linear/2
            w_multiplier = 1 - init_gain_index/2           # make w5_init range from w5_linear to w5_linear/2
            
            nonlinear = model.StratifiedNonLinear1(
                tau=linear.tau, 
                process_noise=linear.process_noise, 
                measure_noise=linear.measure_noise, 
                threshold=th, 
                gain=getattr(linear_prefitted, f"w{n_categories-1}")*init_gain_multiplier, 
                sharpness=5,
                **{f"w{cat}": getattr(linear_prefitted, f"w{cat}")*w_multiplier for cat in range(1, n_categories)}
            )
            
            for _ in range(n_loops):
                # Focus on the gain and the threshold
                nonlinear.fit(
                    state_series=state_train_constant_stim,
                    input_series=input_train_constant_stim,
                    init_params=[getattr(nonlinear, pname) for pname in nonlinear._param_names],
                    bounds=[(1,25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0, 0.5), (0, 2), (0, 10)],
                    fixed_params=['tau', 'process_noise', 'measure_noise', 'sharpness'] + [f'w{cat}' for cat in range(7)],
                    feedback=False
                )
                
                # Focus on the w for all categories except 0
                for cat in range(1, n_categories):

                    nonlinear_one_cat = model.NonLinear1(
                        tau=nonlinear.tau, 
                        process_noise=nonlinear.process_noise, 
                        measure_noise=nonlinear.measure_noise,
                        input_weight=getattr(nonlinear, f'w{cat}'),
                        gain=nonlinear.gain,
                        threshold=nonlinear.threshold, 
                        sharpness=5
                    )
                    
                    nonlinear_one_cat.fit(
                        state_series={0: state_train_constant_stim[cat]},
                        input_series={0: input_train_constant_stim[cat]},
                        init_params=[getattr(nonlinear_one_cat, pname) for pname in nonlinear_one_cat._param_names],
                        bounds=[(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2), (0, 10)],
                        fixed_params=['tau', 'process_noise', 'measure_noise', 'threshold', 'gain', 'sharpness'],
                        feedback=False
                    )
                    nonlinear.set_params({f'w{cat}': nonlinear_one_cat.input_weight})
                    
                # Re-adjust tau
                nonlinear.fit(
                    state_series=state_train_constant_stim,
                    input_series=input_train_constant_stim,
                    init_params=[getattr(nonlinear, pname) for pname in nonlinear._param_names],
                    bounds=[(1,25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0, 0.5), (0, 2), (0, 10)],
                    fixed_params=['threshold', 'process_noise', 'measure_noise', 'sharpness', 'gain'] + [f'w{cat}' for cat in range(7)],
                    feedback=False
                )
            all_models.append(nonlinear)
            ll_models.append(nonlinear.loglikelihood(state_train_constant_stim, input_train_constant_stim))
    
    nonlinear_fitted = all_models[np.argmax(ll_models)]
    
    return nonlinear_fitted



def clever_fit_nonlinear2(linear_prefitted: model.StratifiedLinear, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedNonLinear2:
    """
    Clever fitting strategy for StratifiedNonLinear2 model.
    
    Strategy:
    1. Initialize from prefitted linear model
    2. Grid search over threshold values
    3. For each threshold, alternate between:
       - Fitting gain and threshold
       - Fitting input weights for each category
    """
    n_categories = len(list(state_train.keys()))

    # Fit the OU process on resting-state dynamics
    linear = model.StratifiedLinear(tau=10, process_noise=0.1, measure_noise=0.1)
    linear.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1] + [0]*7,
        bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7,
        fixed_params=[f'w{i}' for i in range(7)])

    # Isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    state_train_constant_stim[0] = state_train[0]
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim[0] = input_train[0]

    # Fit nonlinear (different initializations for the threshold)
    all_models, ll_models = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(2):
            init_gain_multiplier = init_gain_index/2       # make gain_init range from 0 to w5_linear/2
            w_multiplier = 1 - init_gain_index/2           # make w5_init range from w5_linear to w5_linear/2
            
            nonlinear = model.StratifiedNonLinear2(
                tau=linear.tau, 
                process_noise=linear.process_noise, 
                measure_noise=linear.measure_noise, 
                threshold=th, 
                a=0.001,
                b=getattr(linear_prefitted, f"w{n_categories-1}")*init_gain_multiplier, 
                sharpness=5,
                **{f"w{cat}": getattr(linear_prefitted, f"w{cat}")*w_multiplier for cat in range(1, n_categories)}
            )
            
            for _ in range(n_loops):
                # Adjust the threshold
                nonlinear.fit(
                    state_series=state_train_constant_stim,
                    input_series=input_train_constant_stim,
                    init_params=[getattr(nonlinear, pname) for pname in nonlinear._param_names],
                    bounds=[(1,25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0,1), (0, 0.5), (0, 2), (0, 10)],
                    fixed_params=['tau', 'process_noise', 'measure_noise', 'sharpness', 'a', 'b'] + [f'w{cat}' for cat in range(7)],
                    feedback=False
                )
                
                # Focus on the gain parameters: a and b
                nonlinear.fit(
                    state_series=state_train_constant_stim,
                    input_series=input_train_constant_stim,
                    init_params=[getattr(nonlinear, pname) for pname in nonlinear._param_names],
                    bounds=[(1,25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0,1), (0, 0.5), (0, 2), (0, 10)],
                    fixed_params=['tau', 'process_noise', 'measure_noise', 'sharpness', 'threshold'] + [f'w{cat}' for cat in range(7)],
                    feedback=False
                )
                
                # Focus on the w for all categories except 0
                for cat in range(1, n_categories):
                    nonlinear_one_cat = model.NonLinear2(
                        tau=nonlinear.tau, 
                        process_noise=nonlinear.process_noise, 
                        measure_noise=nonlinear.measure_noise,
                        input_weight=getattr(nonlinear, f'w{cat}'),
                        a=nonlinear.a,
                        b=nonlinear.b,
                        threshold=nonlinear.threshold, 
                        sharpness=5
                    )
                    
                    nonlinear_one_cat.fit(
                        state_series={0: state_train_constant_stim[cat]},
                        input_series={0: input_train_constant_stim[cat]},
                        init_params=[getattr(nonlinear_one_cat, pname) for pname in nonlinear_one_cat._param_names],
                        bounds=[(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 0.5), (0, 2), (0, 10)],
                        fixed_params=['tau', 'process_noise', 'measure_noise', 'threshold', 'a', 'b', 'sharpness'],
                        feedback=False
                    )
                    nonlinear.set_params({f'w{cat}': nonlinear_one_cat.input_weight})
                    
                # Re-adjust tau
                nonlinear.fit(
                    state_series=state_train_constant_stim,
                    input_series=input_train_constant_stim,
                    init_params=[getattr(nonlinear, pname) for pname in nonlinear._param_names],
                    bounds=[(1,25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0, 0.5), (0, 2), (0, 10)],
                    fixed_params=['threshold', 'process_noise', 'measure_noise', 'sharpness', 'a', 'b'] + [f'w{cat}' for cat in range(7)],
                    feedback=False
                )
            all_models.append(nonlinear)
            ll_models.append(nonlinear.loglikelihood(state_train_constant_stim, input_train_constant_stim))
    
    nonlinear_fitted = all_models[np.argmax(ll_models)]
    
    return nonlinear_fitted