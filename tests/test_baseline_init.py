"""Stationary initialization of the filter on the baseline category (default)."""
import numpy as np
import pytest

from leadyna import LatentSeries, use_baseline_init
from leadyna.fitting_tools import clever_fit_linear
from leadyna.model import (BaseLEADModel, LinearLEAD, SigmoidFeedbackLEAD,
                           StratifiedGainModulationLEAD, StratifiedSigmoidFeedbackLEAD)

MODELS = {
    "linear": lambda: LinearLEAD(6.0, 0.4, 0.3, n_categories=3, w1=0.05, w2=0.15),
    "fixed_gain": lambda: StratifiedSigmoidFeedbackLEAD(
        6.0, 0.4, 0.3, gain=0.12, threshold=1.0, sharpness=5, n_categories=3, w1=0.05, w2=0.15),
    "gain_mod": lambda: StratifiedGainModulationLEAD(
        6.0, 0.4, 0.3, threshold=1.0, sharpness=5, n_categories=3,
        w1=0.05, w2=0.15, g1=0.1, g2=0.2),
}


def _data(m, n=12, rest_len=40, win=20, seed=0):
    np.random.seed(seed)
    box = np.r_[np.zeros(60), np.ones(win)]
    sim = m.measure_simulations({c: np.tile(box * (c > 0), (n, 1)) for c in range(3)})
    s = {0: sim[0][:, -rest_len:], 1: sim[1][:, -win:], 2: sim[2][:, -win:]}
    return LatentSeries(s, {c: (np.ones_like(v) if c else np.zeros_like(v)) for c, v in s.items()})


def test_default_is_stationary():
    assert BaseLEADModel.baseline_init == "stationary"
    with use_baseline_init("observation"):
        assert MODELS["linear"]().baseline_init == "observation"
    assert BaseLEADModel.baseline_init == "stationary"
    with pytest.raises(ValueError, match="baseline_init"), use_baseline_init("zero"):
        pass


def test_linear_stationary_moments_are_exact():
    m = MODELS["linear"]()
    a, Q = 1 - 1 / m.tau, m.process_noise**2
    mean, var = m.stationary_moments()
    assert abs(mean) < 1e-12
    assert var == pytest.approx(Q / (1 - a * a), rel=1e-10)


@pytest.mark.parametrize("mode", ["stationary", "observation"])
def test_ukf_matches_exact_kalman(mode):
    m = MODELS["linear"]()
    d = _data(m)
    assert m.loglikelihood(d, baseline_init=mode) == pytest.approx(
        m.loglikelihood_kalman(d, baseline_init=mode), abs=1e-8)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("mode", ["stationary", "observation"])
def test_fast_matches_filterpy(name, mode):
    m = MODELS[name]()
    d = _data(m, n=4)
    ref = m.loglikelihood(d, engine="filterpy", n_jobs=1, baseline_init=mode)
    assert m.loglikelihood(d, engine="fast", baseline_init=mode) == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("name", list(MODELS))
def test_only_baseline_category_changes(name):
    m = MODELS[name]()
    d = _data(m)
    stim = LatentSeries({c: d.states[c] for c in (1, 2)}, {c: d.inputs[c] for c in (1, 2)})
    assert m.loglikelihood(stim, baseline_init="stationary") == \
        m.loglikelihood(stim, baseline_init="observation")
    rest = LatentSeries({0: d.states[0]}, {0: d.inputs[0]})
    assert m.loglikelihood(rest, baseline_init="stationary") != \
        m.loglikelihood(rest, baseline_init="observation")


def test_single_category_models_are_untouched():
    one = SigmoidFeedbackLEAD(6.0, 0.4, 0.3, input_weight=0.1, gain=0.1, threshold=1, sharpness=5)
    assert one.baseline_category is None
    d = _data(MODELS["linear"]())
    stim_as_0 = LatentSeries({0: d.states[2]}, {0: d.inputs[2]})
    assert one.loglikelihood(stim_as_0, baseline_init="stationary") == \
        one.loglikelihood(stim_as_0, baseline_init="observation")


def test_short_segments_no_longer_bias_tau():
    truth = LinearLEAD(8.0, 0.4, 0.4, n_categories=2, w1=0.1)
    np.random.seed(1)
    sim = truth.measure_simulations({0: np.zeros((300, 150)), 1: np.ones((300, 120))})
    rest = sim[0][:, 100:]
    segs = np.concatenate([rest[:, k * 25:(k + 1) * 25] for k in range(2)])
    s = {0: segs, 1: sim[1][:, 100:]}
    d = LatentSeries(s, {0: np.zeros_like(s[0]), 1: np.ones_like(s[1])})
    tau_stat = clever_fit_linear(d).tau
    with use_baseline_init("observation"):
        tau_obs = clever_fit_linear(d).tau
    assert abs(tau_stat / 8.0 - 1) < 0.06
    assert tau_obs < 0.9 * tau_stat
