"""Oracle test for the UKF engine, plus the LatentSeries routing equivalence.

For the linear model there are two independent likelihood paths: the UKF
(``loglikelihood``) and the exact analytic scalar Kalman filter
(``loglikelihood_kalman``). They must agree to numerical tolerance — this single
check validates the whole UKF engine for the linear case.
"""
import numpy as np

from leadyna import LatentSeries
from leadyna.model import StratifiedLinear


def _toy_linear_data(seed=0, n_trials=8, n_t=40):
    rng = np.random.default_rng(seed)
    states = {
        0: rng.standard_normal((n_trials, n_t)),
        1: rng.standard_normal((n_trials, n_t)),
    }
    inputs = {
        0: np.zeros((n_trials, n_t)),
        1: np.ones((n_trials, n_t)),
    }
    return states, inputs


def _model():
    return StratifiedLinear(tau=10.0, process_noise=0.2, measure_noise=0.3, w1=0.5)


def test_ukf_matches_exact_kalman():
    states, inputs = _toy_linear_data()
    m = _model()
    ukf = m.loglikelihood(states, inputs, n_jobs=1)
    kalman = m.loglikelihood_kalman(states, inputs)
    assert np.isclose(ukf, kalman, rtol=1e-9, atol=1e-6), (ukf, kalman)


def test_latentseries_routes_like_dicts():
    states, inputs = _toy_linear_data()
    m = _model()
    ll_dicts = m.loglikelihood(states, inputs, n_jobs=1)
    ls = LatentSeries(states, inputs)  # dt defaults to 1.0 (index units)
    ll_ls = m.loglikelihood(ls, n_jobs=1)
    assert np.isclose(ll_dicts, ll_ls, rtol=0, atol=0)


def test_latentseries_kalman_routes_like_dicts():
    states, inputs = _toy_linear_data()
    m = _model()
    ll_dicts = m.loglikelihood_kalman(states, inputs)
    ls = LatentSeries(states, inputs)
    ll_ls = m.loglikelihood_kalman(ls)
    assert np.isclose(ll_dicts, ll_ls, rtol=0, atol=0)
