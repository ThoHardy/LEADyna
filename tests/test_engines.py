"""The fast (vectorized) likelihood engine reproduces the filterpy engine."""
import numpy as np
import pytest

from leadyna import LatentSeries, use_engine
from leadyna.model import (
    AffineFeedbackLEAD,
    BaseLEADModel,
    GainModulationLEAD,
    LinearLEAD,
    SigmoidFeedbackLEAD,
    StratifiedAffineFeedbackLEAD,
    StratifiedGainModulationLEAD,
    StratifiedSigmoidFeedbackLEAD,
)

NOISE = dict(tau=8.0, process_noise=0.25, measure_noise=0.2)
W = dict(w1=0.06, w2=0.12, w3=0.2)
MODELS = {
    "linear": lambda: LinearLEAD(**NOISE, n_categories=4, **W),
    "sigmoid": lambda: SigmoidFeedbackLEAD(**NOISE, input_weight=0.1, gain=0.15,
                                           threshold=1.0, sharpness=5),
    "strat_sigmoid": lambda: StratifiedSigmoidFeedbackLEAD(**NOISE, gain=0.15, threshold=1.0,
                                                           sharpness=5, n_categories=4, **W),
    "affine": lambda: AffineFeedbackLEAD(**NOISE, input_weight=0.1, a=0.05, b=0.1,
                                         threshold=1.0, sharpness=5),
    "strat_affine": lambda: StratifiedAffineFeedbackLEAD(**NOISE, a=0.05, b=0.1, threshold=1.0,
                                                         sharpness=5, n_categories=4, **W),
    "gainmod": lambda: GainModulationLEAD(**NOISE, input_weight=0.1, gain=0.15,
                                          threshold=1.0, sharpness=5),
    "strat_gainmod": lambda: StratifiedGainModulationLEAD(**NOISE, threshold=1.0, sharpness=5,
                                                          n_categories=4, g1=0.05, g2=0.1,
                                                          g3=0.2, **W),
}


def _data(trial_varying=False, seed=0):
    rng = np.random.default_rng(seed)
    states, inputs = {}, {}
    for cat, (n, T) in {0: (12, 20), 1: (9, 30), 2: (7, 30), 3: (10, 25)}.items():
        u = np.zeros((n, T)) if cat == 0 else np.ones((n, T))
        if trial_varying and cat:
            u[:, : T // 2] = rng.choice([0.0, 0.5, 1.0], size=(n, T // 2))
        states[cat] = np.cumsum(rng.normal(0.1 * (cat > 0), 0.3, size=(n, T)), axis=1)
        inputs[cat] = u
    return LatentSeries(states, inputs)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("trial_varying", [False, True])
@pytest.mark.parametrize("vector_inputs", [True, False])
def test_fast_matches_filterpy(name, trial_varying, vector_inputs):
    m = MODELS[name]()
    m.vector_inputs = vector_inputs   # False exercises the scalar-u grouping fallback
    data = _data(trial_varying)
    ref = m.loglikelihood(data, n_jobs=1, engine="filterpy")
    fast = m.loglikelihood(data, engine="fast")
    assert np.isclose(fast, ref, rtol=1e-12, atol=1e-9), (fast, ref)


def test_fast_matches_exact_kalman():
    m = MODELS["linear"]()
    data = _data()
    assert np.isclose(m.loglikelihood(data), m.loglikelihood_kalman(data), rtol=1e-12)


def test_engine_selection():
    m = MODELS["strat_gainmod"]()
    data = _data()
    assert BaseLEADModel.engine == "fast"
    with use_engine("filterpy"):
        assert m.engine == "filterpy"
        assert MODELS["linear"]().engine == "filterpy"
    assert m.engine == "fast"
    with pytest.raises(RuntimeError), use_engine("filterpy"):
        raise RuntimeError
    assert BaseLEADModel.engine == "fast"
    m.engine = "filterpy"
    assert MODELS["strat_gainmod"]().engine == "fast"
    with pytest.raises(ValueError, match="engine"):
        m.loglikelihood(data, engine="jax")
    with pytest.raises(ValueError, match="engine"), use_engine("numba"):
        pass


def test_fit_engines_agree():
    truth = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05, n_categories=2, w1=0.5)
    np.random.seed(2)
    inputs = {1: np.ones((6, 60))}
    states = truth.measure_simulations(inputs)
    kw = dict(init_params=[10.0, 0.1, 0.05, 0.0, 0.0],
              bounds=[(1, 25), (0.01, 1), (0.01, 1), (0, 1), (0, 1)],
              fixed_params=["tau", "process_noise", "measure_noise", "w0"])
    fits = {}
    for engine in ["fast", "filterpy"]:
        m = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05, n_categories=2)
        m.fit(states, inputs, n_jobs=1, engine=engine, **kw)
        fits[engine] = m.w1
    assert abs(fits["fast"] - fits["filterpy"]) < 1e-5
    assert abs(fits["fast"] - 0.5) < 0.15


def test_gainmod_simulation_uses_each_trials_input():
    m = StratifiedGainModulationLEAD(tau=8.0, process_noise=0.0, measure_noise=0.0,
                                     threshold=0.5, sharpness=5, n_categories=4,
                                     g1=0.3, g2=0.6, g3=0.9, w3=0.2)
    rng = np.random.default_rng(3)
    u = rng.choice([0.0, 1.0], size=(6, 30))
    batch = m.measure_simulations({3: u}, initial_states={3: 1.0})[3]
    for i in range(6):
        single = m.measure_simulations({3: u[i:i + 1]}, initial_states={3: 1.0})[3]
        np.testing.assert_allclose(batch[i], single[0])


def test_gainmod_rejects_out_of_range_inputs():
    m = MODELS["strat_gainmod"]()
    data = _data()
    bad = LatentSeries(data.states, {c: v * (-1.0 if c == 2 else 1.0)
                                     for c, v in data.inputs.items()})
    with pytest.raises(ValueError, match="out of range"):
        m.loglikelihood(bad)
