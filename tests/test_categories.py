"""Phase 3: the core is category-count-agnostic.

Free ``n_categories`` must (a) reproduce the historical 7-category model exactly
when asked for 7, (b) run the full UKF==Kalman oracle for a non-7 count, (c)
recover a weight by fitting, and (d) expose ``baseline_category`` and validate
stray weights.
"""
import warnings

import numpy as np
import pytest

from leadyna import LatentSeries
from leadyna.model import LinearLEAD


def _linear_data(n_categories, seed=0, n_trials=6, n_t=40):
    rng = np.random.default_rng(seed)
    states, inputs = {}, {}
    for c in range(n_categories):
        states[c] = rng.standard_normal((n_trials, n_t))
        inputs[c] = np.zeros((n_trials, n_t)) if c == 0 else np.ones((n_trials, n_t))
    return states, inputs


def test_default_matches_explicit_seven():
    m_default = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.1, w1=0.5)
    m_seven = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.1,
                         n_categories=7, w1=0.5)
    assert m_default._param_names == m_seven._param_names
    assert m_default.get_params() == m_seven.get_params()


def test_param_names_track_n_categories():
    m = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.1, n_categories=3)
    assert m._param_names == ["tau", "process_noise", "measure_noise", "w0", "w1", "w2"]
    assert not hasattr(m, "w3")


def test_ukf_matches_kalman_three_categories():
    states, inputs = _linear_data(n_categories=3)
    m = LinearLEAD(tau=10.0, process_noise=0.2, measure_noise=0.3,
                   n_categories=3, w1=0.5, w2=0.8)
    ukf = m.loglikelihood(states, inputs, n_jobs=1)
    kalman = m.loglikelihood_kalman(states, inputs)
    assert np.isclose(ukf, kalman, rtol=1e-9, atol=1e-6), (ukf, kalman)


def test_latentseries_roundtrip_non_seven():
    states, inputs = _linear_data(n_categories=4)
    m = LinearLEAD(tau=10.0, process_noise=0.2, measure_noise=0.3, n_categories=4, w2=0.3)
    ll_dicts = m.loglikelihood(states, inputs, n_jobs=1)
    ll_ls = m.loglikelihood(LatentSeries(states, inputs), n_jobs=1)
    assert np.isclose(ll_dicts, ll_ls, rtol=0, atol=0)


def test_fit_recovers_weight_three_categories():
    np.random.seed(0)
    truth = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05,
                       n_categories=3, w1=0.5)
    inputs = {1: np.ones((8, 80))}
    states = truth.measure_simulations(inputs)
    m = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05, n_categories=3)
    m.fit(
        states, inputs,
        init_params=[10.0, 0.1, 0.05] + [0.0] * 3,
        bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)] * 3,
        fixed_params=["tau", "process_noise", "measure_noise", "w0", "w2"],
        n_jobs=1,
    )
    assert abs(m.w1 - 0.5) < 0.15, m.w1


def test_baseline_category_exposed_and_overridable():
    m = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.1, n_categories=4)
    assert m.baseline_category == 0
    m2 = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.1,
                    n_categories=4, baseline_category=3)
    assert m2.baseline_category == 3


def test_stray_weight_rejected():
    with pytest.raises(TypeError):
        LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.1, n_categories=3, w5=0.1)
