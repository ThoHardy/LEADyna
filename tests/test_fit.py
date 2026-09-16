"""Fitting oracles: parameter recovery, and dict/LatentSeries route equivalence.

All fits pass ``n_jobs=1`` so the tests stay fast and process-spawn-free — the
same knob users should reach for when fitting small data locally.
"""
import numpy as np

from leadyna import LatentSeries
from leadyna.model import LinearLEAD

_FIT = dict(
    init_params=[10.0, 0.1, 0.05] + [0.0] * 7,
    bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)] * 7,
    fixed_params=["tau", "process_noise", "measure_noise"]
    + [f"w{k}" for k in range(7) if k != 1],
    n_jobs=1,
)


def _simulate(w1, n_trials, n_t, seed):
    np.random.seed(seed)
    truth = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05, w1=w1)
    inputs = {1: np.ones((n_trials, n_t))}
    states = truth.measure_simulations(inputs)
    return states, inputs


def test_fit_recovers_linear_weight():
    states, inputs = _simulate(w1=0.5, n_trials=8, n_t=80, seed=0)
    m = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05)
    m.fit(states, inputs, **_FIT)
    assert abs(m.w1 - 0.5) < 0.15, m.w1


def test_fit_latentseries_matches_dicts():
    states, inputs = _simulate(w1=0.4, n_trials=5, n_t=60, seed=1)
    m1 = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05)
    m1.fit(states, inputs, **_FIT)
    m2 = LinearLEAD(tau=10.0, process_noise=0.1, measure_noise=0.05)
    m2.fit(LatentSeries(states, inputs), **_FIT)
    assert np.isclose(m1.w1, m2.w1)
