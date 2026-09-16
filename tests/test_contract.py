"""Validation tests for the LatentSeries contract.

A bad LatentSeries must fail fast with a clear error at construction, not with
an obscure one deep inside the UKF.
"""
import numpy as np
import pytest

from leadyna import LatentSeries
from leadyna.model import StratifiedLinear


def _good():
    states = {0: np.zeros((3, 10)), 1: np.ones((3, 10))}
    inputs = {0: np.zeros((3, 10)), 1: np.ones((3, 10))}
    return states, inputs


def test_valid_construction():
    s, i = _good()
    ls = LatentSeries(s, i, dt=0.01, category_labels={0: "rest", 1: "snr1"})
    assert ls.categories == [0, 1]
    assert ls.n_trials(0) == 3
    assert ls.n_timesteps(1) == 10
    assert len(ls) == 2
    assert ls.dt == 0.01


def test_lists_are_coerced_to_arrays():
    ls = LatentSeries({0: [[0, 1, 2], [3, 4, 5]]}, {0: [[0, 0, 0], [1, 1, 1]]})
    assert isinstance(ls.states[0], np.ndarray)
    assert ls.states[0].shape == (2, 3)


def test_mismatched_categories():
    s, i = _good()
    del i[1]
    with pytest.raises(ValueError):
        LatentSeries(s, i)


def test_shape_mismatch():
    s, i = _good()
    i[0] = np.zeros((3, 9))
    with pytest.raises(ValueError):
        LatentSeries(s, i)


def test_single_timestep():
    with pytest.raises(ValueError):
        LatentSeries({0: np.zeros((3, 1))}, {0: np.zeros((3, 1))})


def test_one_dimensional_array():
    with pytest.raises(ValueError):
        LatentSeries({0: np.zeros(10)}, {0: np.zeros(10)})


def test_empty_states():
    with pytest.raises(ValueError):
        LatentSeries({}, {})


def test_bad_dt_value():
    s, i = _good()
    with pytest.raises(ValueError):
        LatentSeries(s, i, dt=0)
    with pytest.raises(ValueError):
        LatentSeries(s, i, dt=-1.0)


def test_bad_dt_type():
    s, i = _good()
    with pytest.raises(TypeError):
        LatentSeries(s, i, dt="fast")


def test_nonfinite_values_rejected():
    s, i = _good()
    s[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        LatentSeries(s, i)


def test_category_labels_extra_key():
    s, i = _good()
    with pytest.raises(ValueError):
        LatentSeries(s, i, category_labels={9: "nope"})


def test_model_rejects_both_latentseries_and_inputs():
    s, i = _good()
    ls = LatentSeries(s, i)
    m = StratifiedLinear(tau=10.0, process_noise=0.2, measure_noise=0.3)
    with pytest.raises(TypeError):
        m.loglikelihood(ls, i)


def test_model_requires_inputs_for_plain_dict():
    s, _ = _good()
    m = StratifiedLinear(tau=10.0, process_noise=0.2, measure_noise=0.3)
    with pytest.raises(TypeError):
        m.loglikelihood(s)
