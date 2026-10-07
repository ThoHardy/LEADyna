"""Multi-step ("clever") fitting strategies for the LEAD models.

Each ``clever_fit_*`` breaks a high-dimensional MLE into a sequence of small fits
(resting-state OU first, then per-category weights, then the non-linear terms),
with several initialisations, and keeps the best-likelihood solution.

IMPORTANT — difference with LEAD (the prototype toolbox)
--------------------------------------------------------
In LEAD, every ``clever_fit_*`` received the *whole* epochs and cut the
constant-input window itself, with ``input_start_index=75, input_stop_index=100``
as silent defaults (250-500 ms in Sergent et al. 2021 at 100 Hz from -500 ms).

In LEADyna they receive data that are **already windowed** by the user: category
0 holds the baseline segments, every other category holds only the window in
which the input ``I(t)`` is assumed (and fitted). Nothing is cut here anymore,
because choosing that window is choosing the input hypothesis — a scientific
decision that belongs to the user (see the README tutorial).

For backward compatibility, passing ``input_start_index`` / ``input_stop_index``
still reproduces the LEAD behaviour (stimulus categories are sliced
``[:, start:stop]``, category 0 is left whole) with a ``DeprecationWarning``.
Calls that relied on the old *defaults* (75, 100) without passing them now fit
the data as given.

Other conventions
-----------------
* Category 0 is the baseline (resting / noise-only) category. This is fixed
  (``BASELINE_CATEGORY``), matching the EEG frontend's output.
* Category keys need not be contiguous: a subject with no trial in some level
  simply has no key for it; the matching weight stays at 0.
* Bounds, the threshold grid and the sigmoid sharpness are options; their
  defaults are the values used in SOUNDMODEL. Bounds are in the units of the
  data's ``dt`` (index units by default: ``tau`` in (1, 25) samples).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from . import model as model
from .datasets import LatentSeries

__all__ = [
    "BASELINE_CATEGORY",
    "DEFAULT_BOUNDS",
    "DEFAULT_THRESHOLD_GRID",
    "DEFAULT_SHARPNESS",
    "clever_fit_null",
    "clever_fit_linear",
    "clever_fit_gainmodul",
    "clever_fit_nonlinear1",
    "clever_fit_nonlinear2",
]

BASELINE_CATEGORY = 0

DEFAULT_BOUNDS = {
    "tau": (1, 25),
    "process_noise": (0.01, 1),
    "measure_noise": (0.01, 1),
    "w": (0, 1),           # every per-category input weight w0..wN (and input_weight)
    "g": (0, 1),           # every per-category gain g0..gN (gain-modulated model)
    "gain": (0, 0.5),      # shared gain of the fixed-gain model
    "threshold": (0, 2),
    "sharpness": (0, 10),
    "a": (0, 1),
    "b": (0, 0.5),
}

DEFAULT_THRESHOLD_GRID = tuple(np.linspace(0, 2, 5))
DEFAULT_SHARPNESS = 5

_INIT_NOISE = [10, 0.1, 0.1]  # tau, process_noise, measure_noise starting point


# =============================================================================
# Internal helpers
# =============================================================================

@dataclass
class _Data:
    base_s: np.ndarray
    base_i: np.ndarray
    stim_s: dict
    stim_i: dict
    stim_cats: list
    n_categories: int
    dt: float

    def ls(self, states, inputs):
        return LatentSeries(states, inputs, dt=self.dt)

    def baseline(self):
        return self.ls({BASELINE_CATEGORY: self.base_s}, {BASELINE_CATEGORY: self.base_i})

    def stim(self, cats=None):
        cats = self.stim_cats if cats is None else cats
        return self.ls({c: self.stim_s[c] for c in cats}, {c: self.stim_i[c] for c in cats})

    def all(self):
        # Stimulus categories first, baseline last: same order as LEAD (sum order matters
        # at the 1e-16 level, so this keeps fits bit-identical to the prototype).
        return self.ls({**self.stim_s, BASELINE_CATEGORY: self.base_s},
                       {**self.stim_i, BASELINE_CATEGORY: self.base_i})


def _prepare(state_train, input_train, start, stop) -> _Data:
    if isinstance(state_train, LatentSeries):
        if input_train is not None:
            raise TypeError("Pass a LatentSeries alone, or two dicts — not both.")
        if state_train.inputs is None:
            raise ValueError(
                "This LatentSeries has no inputs: attach your input hypothesis with "
                "latent.with_inputs(inputs) before fitting."
            )
        states, inputs, dt = state_train.states, state_train.inputs, state_train.dt
    else:
        if input_train is None:
            raise TypeError("input_train is required when state_train is a dict.")
        states, inputs, dt = state_train, input_train, 1.0

    if BASELINE_CATEGORY not in states:
        raise ValueError(
            f"Category {BASELINE_CATEGORY} (baseline / resting state) is required; "
            f"got categories {sorted(states)}."
        )
    stim_cats = sorted(c for c in states if c != BASELINE_CATEGORY)
    if not stim_cats:
        raise ValueError("At least one stimulus category (key != 0) is required.")

    if start is not None or stop is not None:
        warnings.warn(
            "input_start_index / input_stop_index reproduce the LEAD behaviour (slicing "
            "inside clever_fit_*). In LEADyna, window the stimulus categories yourself "
            "and pass the windowed data.",
            DeprecationWarning,
            stacklevel=3,
        )
        stim_s = {c: states[c][:, start:stop] for c in stim_cats}
        stim_i = {c: inputs[c][:, start:stop] for c in stim_cats}
    else:
        stim_s = {c: states[c] for c in stim_cats}
        stim_i = {c: inputs[c] for c in stim_cats}

    return _Data(
        base_s=states[BASELINE_CATEGORY],
        base_i=inputs[BASELINE_CATEGORY],
        stim_s=stim_s,
        stim_i=stim_i,
        stim_cats=stim_cats,
        n_categories=max(states) + 1,
        dt=dt,
    )


def _merge_bounds(bounds) -> dict:
    merged = dict(DEFAULT_BOUNDS)
    for key, val in (bounds or {}).items():
        if key not in DEFAULT_BOUNDS:
            raise KeyError(f"Unknown bound '{key}'; expected any of {sorted(DEFAULT_BOUNDS)}.")
        merged[key] = tuple(val)
    return merged


def _bounds_for(m, bounds: dict, alias=None) -> list:
    """Bounds list ordered like ``m._param_names``, looked up by parameter family."""
    alias = alias or {}
    out = []
    for name in m._param_names:
        key = alias.get(name, name)
        if key == "input_weight" or re.fullmatch(r"w\d+", key):
            key = "w"
        elif re.fullmatch(r"g\d+", key):
            key = "g"
        out.append(bounds[key])
    return out


def _init(m) -> list:
    return [getattr(m, p) for p in m._param_names]


def _ll(m, data: LatentSeries, n_jobs) -> float:
    return m.loglikelihood(data) if n_jobs is None else m.loglikelihood(data, n_jobs=n_jobs)


def _all_w(n):
    return [f"w{i}" for i in range(n)]


def _fit_resting_ou(d: _Data, bounds, n_jobs) -> model.LinearLEAD:
    """OU process (all weights 0) fitted on the baseline category alone."""
    n = d.n_categories
    rest = model.LinearLEAD(n_categories=n, tau=10, process_noise=0.1, measure_noise=0.1, w0=0)
    rest.fit(d.baseline(), init_params=_INIT_NOISE + [0] * n, bounds=_bounds_for(rest, bounds),
             fixed_params=_all_w(n), n_jobs=n_jobs)
    return rest


# =============================================================================
# Clever fits
# =============================================================================

def clever_fit_null(state_train, input_train=None, input_start_index=None,
                    input_stop_index=None, *, bounds=None, n_jobs=None) -> model.LinearLEAD:
    """Null model: one OU process (``tau``, noises) shared by all categories, all ``w = 0``.

    ``state_train`` is a windowed :class:`LatentSeries` (or a dict, with
    ``input_train``): category 0 = baseline segments, other categories = the
    constant-input window only.
    """
    d = _prepare(state_train, input_train, input_start_index, input_stop_index)
    b, n = _merge_bounds(bounds), d.n_categories
    null = model.LinearLEAD(n_categories=n, tau=10, process_noise=0.1, measure_noise=0.1, w0=0)
    null.fit(d.all(), init_params=_INIT_NOISE + [0] * n, bounds=_bounds_for(null, b),
             fixed_params=_all_w(n), n_jobs=n_jobs)
    return null


def clever_fit_linear(state_train, input_train=None, input_start_index=None,
                      input_stop_index=None, *, bounds=None, n_jobs=None) -> model.LinearLEAD:
    """Linear (OU) model with one input weight per category.

    1. Fit ``tau`` and the noises on the baseline category (all ``w = 0``).
    2. For each stimulus category, fit its weight alone (two initialisations).

    ``state_train`` must already be windowed (see module docstring).
    """
    d = _prepare(state_train, input_train, input_start_index, input_stop_index)
    b, n = _merge_bounds(bounds), d.n_categories
    rest = _fit_resting_ou(d, b, n_jobs)

    linear = model.LinearLEAD(n_categories=n, tau=rest.tau, process_noise=rest.process_noise,
                              measure_noise=rest.measure_noise)
    linear.dt = d.dt  # never fitted directly below, so set its timestep explicitly
    for cat in d.stim_cats:
        one_cat = d.stim([cat])
        ws, lls = [], []
        for w_init in [0, 0.1]:
            m = model.LinearLEAD(n_categories=n, tau=linear.tau,
                                 process_noise=linear.process_noise,
                                 measure_noise=linear.measure_noise)
            setattr(m, f"w{cat}", w_init)
            init = [linear.tau, linear.process_noise, linear.measure_noise] + [0] * n
            init[3 + cat] = w_init
            m.fit(one_cat, init_params=init, bounds=_bounds_for(m, b),
                  fixed_params=["tau", "measure_noise", "process_noise"]
                  + [f"w{i}" for i in range(n) if i != cat],
                  n_jobs=n_jobs)
            ws.append(getattr(m, f"w{cat}"))
            lls.append(_ll(m, one_cat, n_jobs))
        linear.set_params({f"w{cat}": ws[np.argmax(lls)]})
    return linear


def clever_fit_gainmodul(linear_prefitted, state_train, input_train=None, n_loops=2,
                         input_start_index=None, input_stop_index=None, *, bounds=None,
                         threshold_grid=DEFAULT_THRESHOLD_GRID, sharpness=DEFAULT_SHARPNESS,
                         n_jobs=None) -> model.StratifiedGainModulationLEAD:
    """Gain-modulated model (one gain ``g`` per category), warm-started from a linear fit.

    For each initial threshold in ``threshold_grid`` and two gain/weight splits,
    alternate ``n_loops`` times between fitting each category's ``(w, g)`` pair and
    the shared threshold; keep the best-likelihood model. The joint steps use the
    stimulus categories only (as in LEAD). ``state_train`` must already be windowed.
    """
    d = _prepare(state_train, input_train, input_start_index, input_stop_index)
    b, n = _merge_bounds(bounds), d.n_categories
    rest = _fit_resting_ou(d, b, n_jobs)
    stim = d.stim()

    candidates, lls = [], []
    for th in threshold_grid:
        for init_gain_index in range(2):
            gain_mult = init_gain_index / 2
            w_mult = 1 - init_gain_index / 2
            gm = model.StratifiedGainModulationLEAD(
                n_categories=n, tau=rest.tau, process_noise=rest.process_noise,
                measure_noise=rest.measure_noise, threshold=th, sharpness=sharpness,
                **{f"w{c}": getattr(linear_prefitted, f"w{c}") * w_mult for c in d.stim_cats},
                **{f"g{c}": getattr(linear_prefitted, f"w{c}") * gain_mult for c in d.stim_cats},
            )
            for _ in range(n_loops):
                for cat in d.stim_cats:
                    one = model.SigmoidFeedbackLEAD(
                        tau=gm.tau, process_noise=gm.process_noise,
                        measure_noise=gm.measure_noise, input_weight=getattr(gm, f"w{cat}"),
                        gain=getattr(gm, f"g{cat}"), threshold=gm.threshold,
                        sharpness=sharpness,
                    )
                    one.fit(d.ls({0: d.stim_s[cat]}, {0: d.stim_i[cat]}),
                            init_params=_init(one), bounds=_bounds_for(one, b, {"gain": "g"}),
                            fixed_params=["tau", "process_noise", "measure_noise",
                                          "threshold", "sharpness"],
                            n_jobs=n_jobs)
                    gm.set_params({f"w{cat}": one.input_weight, f"g{cat}": one.gain})
                gm.fit(stim, init_params=_init(gm), bounds=_bounds_for(gm, b),
                       fixed_params=["tau", "process_noise", "measure_noise", "sharpness"]
                       + _all_w(n) + [f"g{i}" for i in range(n)],
                       n_jobs=n_jobs)
            candidates.append(gm)
            lls.append(_ll(gm, stim, n_jobs))
    return candidates[np.argmax(lls)]


def clever_fit_nonlinear1(linear_prefitted, state_train, input_train=None, n_loops=2,
                          input_start_index=None, input_stop_index=None, *, bounds=None,
                          threshold_grid=DEFAULT_THRESHOLD_GRID, sharpness=DEFAULT_SHARPNESS,
                          n_jobs=None) -> model.StratifiedSigmoidFeedbackLEAD:
    """Fixed-gain model (one sigmoid feedback shared by all categories).

    For each initial threshold and two gain/weight splits, alternate ``n_loops``
    times between (gain, threshold), the per-category weights, and ``tau``; keep
    the best-likelihood model. ``state_train`` must already be windowed.
    """
    d = _prepare(state_train, input_train, input_start_index, input_stop_index)
    b, n = _merge_bounds(bounds), d.n_categories
    rest = _fit_resting_ou(d, b, n_jobs)
    data = d.all()
    top = d.stim_cats[-1]

    candidates, lls = [], []
    for th in threshold_grid:
        for init_gain_index in range(2):
            gain_mult = init_gain_index / 2
            w_mult = 1 - init_gain_index / 2
            nl = model.StratifiedSigmoidFeedbackLEAD(
                n_categories=n, tau=rest.tau, process_noise=rest.process_noise,
                measure_noise=rest.measure_noise, threshold=th,
                gain=getattr(linear_prefitted, f"w{top}") * gain_mult, sharpness=sharpness,
                **{f"w{c}": getattr(linear_prefitted, f"w{c}") * w_mult for c in d.stim_cats},
            )
            for _ in range(n_loops):
                nl.fit(data, init_params=_init(nl), bounds=_bounds_for(nl, b),
                       fixed_params=["tau", "process_noise", "measure_noise", "sharpness"]
                       + _all_w(n),
                       n_jobs=n_jobs)
                for cat in d.stim_cats:
                    one = model.SigmoidFeedbackLEAD(
                        tau=nl.tau, process_noise=nl.process_noise,
                        measure_noise=nl.measure_noise, input_weight=getattr(nl, f"w{cat}"),
                        gain=nl.gain, threshold=nl.threshold, sharpness=sharpness,
                    )
                    one.fit(d.ls({0: d.stim_s[cat]}, {0: d.stim_i[cat]}),
                            init_params=_init(one), bounds=_bounds_for(one, b),
                            fixed_params=["tau", "process_noise", "measure_noise",
                                          "threshold", "gain", "sharpness"],
                            n_jobs=n_jobs)
                    nl.set_params({f"w{cat}": one.input_weight})
                nl.fit(data, init_params=_init(nl), bounds=_bounds_for(nl, b),
                       fixed_params=["threshold", "process_noise", "measure_noise",
                                     "sharpness", "gain"] + _all_w(n),
                       n_jobs=n_jobs)
            candidates.append(nl)
            lls.append(_ll(nl, data, n_jobs))
    return candidates[np.argmax(lls)]


def clever_fit_nonlinear2(linear_prefitted, state_train, input_train=None, n_loops=2,
                          input_start_index=None, input_stop_index=None, *, bounds=None,
                          threshold_grid=DEFAULT_THRESHOLD_GRID, sharpness=DEFAULT_SHARPNESS,
                          n_jobs=None) -> model.StratifiedAffineFeedbackLEAD:
    """Affine-feedback model ``(a*x + b) * sigmoid``, warm-started from a linear fit.

    ``state_train`` must already be windowed (see module docstring).
    """
    d = _prepare(state_train, input_train, input_start_index, input_stop_index)
    b_, n = _merge_bounds(bounds), d.n_categories
    rest = _fit_resting_ou(d, b_, n_jobs)
    data = d.all()
    top = d.stim_cats[-1]

    candidates, lls = [], []
    for th in threshold_grid:
        for init_gain_index in range(2):
            gain_mult = init_gain_index / 2
            w_mult = 1 - init_gain_index / 2
            nl = model.StratifiedAffineFeedbackLEAD(
                n_categories=n, tau=rest.tau, process_noise=rest.process_noise,
                measure_noise=rest.measure_noise, threshold=th, a=0.001,
                b=getattr(linear_prefitted, f"w{top}") * gain_mult, sharpness=sharpness,
                **{f"w{c}": getattr(linear_prefitted, f"w{c}") * w_mult for c in d.stim_cats},
            )
            for _ in range(n_loops):
                nl.fit(data, init_params=_init(nl), bounds=_bounds_for(nl, b_),
                       fixed_params=["tau", "process_noise", "measure_noise", "sharpness",
                                     "a", "b"] + _all_w(n),
                       n_jobs=n_jobs)
                nl.fit(data, init_params=_init(nl), bounds=_bounds_for(nl, b_),
                       fixed_params=["tau", "process_noise", "measure_noise", "sharpness",
                                     "threshold"] + _all_w(n),
                       n_jobs=n_jobs)
                for cat in d.stim_cats:
                    one = model.AffineFeedbackLEAD(
                        tau=nl.tau, process_noise=nl.process_noise,
                        measure_noise=nl.measure_noise, input_weight=getattr(nl, f"w{cat}"),
                        a=nl.a, b=nl.b, threshold=nl.threshold, sharpness=sharpness,
                    )
                    one.fit(d.ls({0: d.stim_s[cat]}, {0: d.stim_i[cat]}),
                            init_params=_init(one), bounds=_bounds_for(one, b_),
                            fixed_params=["tau", "process_noise", "measure_noise",
                                          "threshold", "a", "b", "sharpness"],
                            n_jobs=n_jobs)
                    nl.set_params({f"w{cat}": one.input_weight})
                nl.fit(data, init_params=_init(nl), bounds=_bounds_for(nl, b_),
                       fixed_params=["threshold", "process_noise", "measure_noise",
                                     "sharpness", "a", "b"] + _all_w(n),
                       n_jobs=n_jobs)
            candidates.append(nl)
            lls.append(_ll(nl, data, n_jobs))
    return candidates[np.argmax(lls)]
