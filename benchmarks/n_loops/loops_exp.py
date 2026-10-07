"""Goodness of fit of the fixed-gain and gain-modulated clever fits as n_loops grows.

Instrumented copies of clever_fit_nonlinear1 / clever_fit_gainmodul (same steps, same
order) that snapshot every candidate after each loop; the best candidate at loop k is
what the library returns with n_loops=k (checked in check()).
"""
import copy
import json
import sys
import time

import numpy as np

from leadyna import LatentSeries, fitting_tools as ft, model as lm

SEG, WIN0, WIN1 = 20, 60, 100
W = dict(w1=0.02, w2=0.04, w3=0.06, w4=0.08)
TRUTHS = {
    "GM": lambda: lm.StratifiedGainModulationLEAD(
        tau=10.0, process_noise=0.3, measure_noise=0.3, threshold=1.0, sharpness=5,
        n_categories=5, g1=0.05, g2=0.1, g3=0.15, g4=0.2, **W),
    "FG": lambda: lm.StratifiedSigmoidFeedbackLEAD(
        tau=10.0, process_noise=0.3, measure_noise=0.3, gain=0.15, threshold=1.0,
        sharpness=5, n_categories=5, **W),
}


def simulate(truth_name, n_trials, seed):
    np.random.seed(seed)
    truth = TRUTHS[truth_name]()
    box = np.r_[np.zeros(WIN0), np.ones(WIN1 - WIN0), np.zeros(20)]
    sim = truth.measure_simulations({c: np.tile(box * (c > 0), (n_trials, 1)) for c in range(5)})
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("train", "test"):
        out[split] = ({}, {})
    for c in range(5):
        order = rng.permutation(n_trials)
        n_test = n_trials // 5
        for split, idx in (("test", order[:n_test]), ("train", order[n_test:])):
            x = sim[c][idx]
            if c == 0:
                segs = [x[:, k * SEG:(k + 1) * SEG] for k in range(x.shape[1] // SEG)]
                out[split][0].setdefault(0, []).extend(segs)
            else:
                out[split][0][c] = x[:, WIN0:WIN1]
                out[split][0].setdefault(0, []).append(x[:, :SEG])
    data = {}
    for split, (s, _) in out.items():
        s[0] = np.concatenate(s[0])
        inputs = {c: (np.ones_like(v) if c else np.zeros_like(v)) for c, v in s.items()}
        data[split] = LatentSeries(s, inputs)
    return truth, data["train"], data["test"]


def tracked_gainmodul(lin, data, n_loops, n_thresholds=5, sharpness=5):
    d = ft._prepare(data, None, None, None)
    b, n = ft._merge_bounds(None), d.n_categories
    rest = ft._fit_resting_ou(d, b, None)
    stim = d.stim()
    snaps = [[] for _ in range(n_loops)]
    for th in ft._threshold_inits(None, n_thresholds, b):
        for igi in range(2):
            gm_, wm = igi / 2, 1 - igi / 2
            gm = lm.StratifiedGainModulationLEAD(
                n_categories=n, tau=rest.tau, process_noise=rest.process_noise,
                measure_noise=rest.measure_noise, threshold=th, sharpness=sharpness,
                **{f"w{c}": getattr(lin, f"w{c}") * wm for c in d.stim_cats},
                **{f"g{c}": getattr(lin, f"w{c}") * gm_ for c in d.stim_cats})
            for k in range(n_loops):
                for cat in d.stim_cats:
                    one = lm.SigmoidFeedbackLEAD(
                        tau=gm.tau, process_noise=gm.process_noise, measure_noise=gm.measure_noise,
                        input_weight=getattr(gm, f"w{cat}"), gain=getattr(gm, f"g{cat}"),
                        threshold=gm.threshold, sharpness=sharpness)
                    one.fit(d.ls({0: d.stim_s[cat]}, {0: d.stim_i[cat]}), init_params=ft._init(one),
                            bounds=ft._bounds_for(one, b, {"gain": "g"}),
                            fixed_params=["tau", "process_noise", "measure_noise", "threshold",
                                          "sharpness"])
                    gm.set_params({f"w{cat}": one.input_weight, f"g{cat}": one.gain})
                gm.fit(stim, init_params=ft._init(gm), bounds=ft._bounds_for(gm, b),
                       fixed_params=["tau", "process_noise", "measure_noise", "sharpness"]
                       + ft._all_w(n) + [f"g{i}" for i in range(n)])
                snaps[k].append((gm.loglikelihood(stim), copy.deepcopy(gm)))
    return [max(s, key=lambda t: t[0])[1] for s in snaps]


def tracked_nonlinear1(lin, data, n_loops, sharpness=5):
    d = ft._prepare(data, None, None, None)
    b, n = ft._merge_bounds(None), d.n_categories
    rest = ft._fit_resting_ou(d, b, None)
    full = d.all()
    top = d.stim_cats[-1]
    snaps = [[] for _ in range(n_loops)]
    for th in ft.DEFAULT_THRESHOLD_GRID:
        for igi in range(2):
            gm_, wm = igi / 2, 1 - igi / 2
            nl = lm.StratifiedSigmoidFeedbackLEAD(
                n_categories=n, tau=rest.tau, process_noise=rest.process_noise,
                measure_noise=rest.measure_noise, threshold=th,
                gain=getattr(lin, f"w{top}") * gm_, sharpness=sharpness,
                **{f"w{c}": getattr(lin, f"w{c}") * wm for c in d.stim_cats})
            for k in range(n_loops):
                nl.fit(full, init_params=ft._init(nl), bounds=ft._bounds_for(nl, b),
                       fixed_params=["tau", "process_noise", "measure_noise", "sharpness"]
                       + ft._all_w(n))
                for cat in d.stim_cats:
                    one = lm.SigmoidFeedbackLEAD(
                        tau=nl.tau, process_noise=nl.process_noise, measure_noise=nl.measure_noise,
                        input_weight=getattr(nl, f"w{cat}"), gain=nl.gain,
                        threshold=nl.threshold, sharpness=sharpness)
                    one.fit(d.ls({0: d.stim_s[cat]}, {0: d.stim_i[cat]}), init_params=ft._init(one),
                            bounds=ft._bounds_for(one, b),
                            fixed_params=["tau", "process_noise", "measure_noise", "threshold",
                                          "gain", "sharpness"])
                    nl.set_params({f"w{cat}": one.input_weight})
                nl.fit(full, init_params=ft._init(nl), bounds=ft._bounds_for(nl, b),
                       fixed_params=["threshold", "process_noise", "measure_noise", "sharpness",
                                     "gain"] + ft._all_w(n))
                snaps[k].append((nl.loglikelihood(full), copy.deepcopy(nl)))
    return [max(s, key=lambda t: t[0])[1] for s in snaps]


def polish(m, data):
    """Joint L-BFGS-B over every free parameter (sharpness and baseline weights fixed)."""
    m = copy.deepcopy(m)
    b = ft._merge_bounds(None)
    fixed = ["sharpness", "w0"] + (["g0"] if hasattr(m, "g0") else [])
    m.fit(data, init_params=ft._init(m), bounds=ft._bounds_for(m, b), fixed_params=fixed)
    return m


def param_error(m, truth):
    keys = [k for k in truth.get_params() if k != "sharpness"]
    return {k: float(getattr(m, k) - getattr(truth, k)) for k in keys}


def n_samples(ls):
    return sum(v.size - v.shape[0] for v in ls.states.values())


def check():
    truth, train, test = simulate("GM", 30, 0)
    lin = ft.clever_fit_linear(train)
    tr_gm = tracked_gainmodul(lin, train, 2)
    tr_fg = tracked_nonlinear1(lin, train, 2)
    for k in (1, 2):
        a = ft.clever_fit_gainmodul(lin, train, n_loops=k).get_params()
        b = ft.clever_fit_nonlinear1(lin, train, n_loops=k).get_params()
        assert a == tr_gm[k - 1].get_params(), ("GM", k)
        assert b == tr_fg[k - 1].get_params(), ("FG", k)
    print("tracked fits == library fits for n_loops = 1, 2")


def run(n_trials, seeds, max_loops, out):
    rows = []
    for truth_name in TRUTHS:
        for seed in seeds:
            truth, train, test = simulate(truth_name, n_trials, seed)
            ns_tr, ns_te = n_samples(train), n_samples(test)
            ll_truth = (truth.loglikelihood(train) / ns_tr, truth.loglikelihood(test) / ns_te)
            lin = ft.clever_fit_linear(train)
            for fit_name, tracker in (("GM", tracked_gainmodul), ("FG", tracked_nonlinear1)):
                t0 = time.perf_counter()
                per_loop = tracker(lin, train, max_loops)
                elapsed = time.perf_counter() - t0
                pol = polish(per_loop[-1], train)
                for k, m in enumerate(per_loop, 1):
                    rows.append(dict(truth=truth_name, seed=seed, fit=fit_name, loop=k,
                                     train=m.loglikelihood(train) / ns_tr,
                                     test=m.loglikelihood(test) / ns_te,
                                     truth_train=ll_truth[0], truth_test=ll_truth[1],
                                     err=param_error(m, truth) if fit_name == truth_name else None,
                                     params=m.get_params()))
                rows.append(dict(truth=truth_name, seed=seed, fit=fit_name, loop="polish",
                                 train=pol.loglikelihood(train) / ns_tr,
                                 test=pol.loglikelihood(test) / ns_te,
                                 truth_train=ll_truth[0], truth_test=ll_truth[1],
                                 err=param_error(pol, truth) if fit_name == truth_name else None,
                                 params=pol.get_params(), seconds=elapsed))
                print(truth_name, seed, fit_name, f"{elapsed:.0f}s", flush=True)
                json.dump(rows, open(out, "w"), default=float)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        run(int(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5])
