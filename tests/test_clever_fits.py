"""clever_fit_*: windowed inputs, LEAD equivalence, bounds options, missing categories.

The equivalence tests compare the windowed clever fits with the frozen LEAD code
(bit-identical parameters) under both likelihood engines. Under the fast engine they
take seconds; under the filterpy engine (exact LEAD reproduction) they take ~6 min and
are marked ``slow``: run them with ``pytest -m slow``.
"""
import warnings

import numpy as np
import pytest

import _legacy_fitting_tools as legacy
from leadyna import LatentSeries, fitting_tools
from leadyna.model import BaseLEADModel, StratifiedSigmoidFeedbackLEAD, use_engine

START, STOP = 15, 35


ENGINES = ["fast", pytest.param("filterpy", marks=pytest.mark.slow)]


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    # LEAD code paths call loglikelihood with the filterpy default n_jobs=8; force 1 so
    # the oracle runs fast. Results do not depend on n_jobs.
    monkeypatch.setattr(BaseLEADModel.loglikelihood, "__defaults__", (None, 1, 20, None))


@pytest.fixture(scope="module")
def raw():
    np.random.seed(0)
    truth = StratifiedSigmoidFeedbackLEAD(
        tau=8.0, process_noise=0.15, measure_noise=0.1, gain=0.15, threshold=1.0,
        sharpness=5, n_categories=3, w1=0.05, w2=0.15,
    )
    one = np.r_[np.zeros(START), np.ones(STOP - START), np.zeros(10)]
    inputs = {c: np.tile(one * (c > 0), (8, 1)) for c in range(3)}
    states = truth.measure_simulations(inputs)
    return states, inputs


def _windowed(states, inputs):
    s = {c: states[c][:, START:STOP] for c in (1, 2)}
    i = {c: inputs[c][:, START:STOP] for c in (1, 2)}
    s[0], i[0] = states[0], inputs[0]
    return LatentSeries(s, i)


def _same(m1, m2):
    p1, p2 = m1.get_params(), m2.get_params()
    assert p1.keys() == p2.keys()
    for k in p1:
        assert p1[k] == p2[k], (k, p1[k], p2[k])


@pytest.mark.parametrize("engine", ENGINES)
def test_linear_and_null_match_lead(raw, engine):
    states, inputs = raw
    w = _windowed(states, inputs)
    with use_engine(engine):
        old_lin = legacy.clever_fit_linear(states, inputs, START, STOP)
        old_null = legacy.clever_fit_null(states, inputs, START, STOP)
        _same(fitting_tools.clever_fit_linear(w, n_jobs=1), old_lin)
        _same(fitting_tools.clever_fit_null(w, n_jobs=1), old_null)
        with pytest.warns(DeprecationWarning, match="input_start_index"):
            legacy_path = fitting_tools.clever_fit_linear(states, inputs, START, STOP)
    _same(legacy_path, old_lin)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["clever_fit_gainmodul", "clever_fit_nonlinear1"])
def test_nonlinear_fits_match_lead(raw, name, engine):
    states, inputs = raw
    kw = dict(n_loops=1)
    with use_engine(engine):
        lin = legacy.clever_fit_linear(states, inputs, START, STOP)
        old = getattr(legacy, name)(lin, states, inputs, input_start_index=START,
                                    input_stop_index=STOP, **kw)
        new = getattr(fitting_tools, name)(lin, _windowed(states, inputs), n_jobs=1, **kw)
    _same(new, old)


def test_default_bounds_reproduce_lead_lists():
    from leadyna.model import StratifiedGainModulationLEAD
    gm = StratifiedGainModulationLEAD(1, 0.1, 0.1, 0.5, 5, n_categories=7)
    got = fitting_tools._bounds_for(gm, fitting_tools.DEFAULT_BOUNDS)
    assert got == [(1, 25), (0.01, 1), (0.01, 1), (0, 2), (0, 10)] + [(0, 1)] * 14


def test_custom_bounds_are_used(raw):
    states, inputs = raw
    lin = fitting_tools.clever_fit_linear(_windowed(states, inputs),
                                          bounds={"tau": (30, 60)}, n_jobs=1)
    assert 30 <= lin.tau <= 60
    with pytest.raises(KeyError, match="taux"):
        fitting_tools.clever_fit_linear(_windowed(states, inputs), bounds={"taux": (1, 2)})


def test_missing_category_keeps_weight_indices(raw):
    states, inputs = raw
    w = _windowed(states, inputs)
    sparse = LatentSeries({0: w.states[0], 2: w.states[2]}, {0: w.inputs[0], 2: w.inputs[2]})
    lin = fitting_tools.clever_fit_linear(sparse, n_jobs=1)
    assert lin.n_categories == 3 and lin.w1 == 0 and lin.w2 > 0


def test_requires_baseline_and_inputs(raw):
    states, inputs = raw
    w = _windowed(states, inputs)
    with pytest.raises(ValueError, match="Category 0"):
        fitting_tools.clever_fit_linear({1: w.states[1]}, {1: w.inputs[1]})
    with pytest.raises(ValueError, match="with_inputs"):
        fitting_tools.clever_fit_linear(LatentSeries(w.states))


def test_dt_from_latentseries_is_used(raw):
    states, inputs = raw
    w = _windowed(states, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = fitting_tools.clever_fit_linear(
            LatentSeries(w.states, w.inputs, dt=0.5), bounds={"tau": (0.5, 50)}, n_jobs=1)
    assert lin.dt == 0.5


def test_gainmodul_n_thresholds(raw, monkeypatch):
    states, inputs = raw
    w = _windowed(states, inputs)
    lin = fitting_tools.clever_fit_linear(w)
    tried = []
    original = fitting_tools.model.StratifiedGainModulationLEAD

    def spy(*args, **kwargs):
        tried.append(kwargs["threshold"])
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting_tools.model, "StratifiedGainModulationLEAD", spy)
    fitting_tools.clever_fit_gainmodul(lin, w, n_loops=1)
    assert tried[::2] == list(np.linspace(0, 2, 5))
    tried.clear()
    fitting_tools.clever_fit_gainmodul(lin, w, n_loops=1, n_thresholds=3,
                                       bounds={"threshold": (0.5, 1.5)})
    assert tried[::2] == [0.5, 1.0, 1.5]
    tried.clear()
    fitting_tools.clever_fit_gainmodul(lin, w, n_loops=1, threshold_grid=(0.7,))
    assert tried == [0.7, 0.7]
    with pytest.raises(ValueError, match="n_thresholds"):
        fitting_tools.clever_fit_gainmodul(lin, w, n_thresholds=0)
