import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from leadyna import LatentSeries, LinearLEAD
from leadyna import fitting_tools as ft

N_CAT, N_TRIALS, T, BURN = 6, 150, 250, 100
PRE, WIN0, WIN1, CHUNK = 50, 75, 100, 50


def draw_params(rng):
    tau = float(np.exp(rng.uniform(np.log(2), np.log(13))))
    pn, mn = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.15, 0.8))
    wmax = float(rng.uniform(0.05, 0.45))
    ws = {f"w{c}": wmax * (c / (N_CAT - 1)) ** 1.5 for c in range(N_CAT)}
    return dict(tau=tau, process_noise=pn, measure_noise=mn, **ws)


def simulate(p, seed):
    np.random.seed(seed)
    truth = LinearLEAD(n_categories=N_CAT, **p)
    box = np.zeros(BURN + T)
    box[BURN + WIN0:BURN + WIN1] = 1
    sim = truth.measure_simulations({c: np.tile(box * (c > 0), (N_TRIALS, 1)) for c in range(N_CAT)})
    return {c: x[:, BURN:] for c, x in sim.items()}


def baseline_correct(sim):
    return {c: x - x[:, :PRE].mean(1, keepdims=True) for c, x in sim.items()}


def build(sim, method):
    s = {c: sim[c][:, WIN0:WIN1] for c in range(1, N_CAT)}
    rest = sim[0]
    if method == "full":
        s[0] = rest
    else:
        segs = [rest[:, k * CHUNK:(k + 1) * CHUNK] for k in range(T // CHUNK)]
        if method == "chunks+pre":
            segs += [sim[c][:, :PRE] for c in range(1, N_CAT)]
        s[0] = np.concatenate(segs)
    inputs = {c: (np.ones_like(v) if c else np.zeros_like(v)) for c, v in s.items()}
    return LatentSeries(s, inputs)


METHODS = ("full", "chunks", "chunks+pre")


def run(seeds, out, variants=("raw", "bc")):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        p = draw_params(rng)
        sim0 = simulate(p, seed)
        for variant in variants:
            sim = baseline_correct(sim0) if variant == "bc" else sim0
            for method in METHODS:
                t0 = time.perf_counter()
                fit = ft.clever_fit_linear(build(sim, method))
                rows.append(dict(seed=seed, variant=variant, method=method, truth=p,
                                 fit={k: float(v) for k, v in fit.get_params().items()},
                                 seconds=time.perf_counter() - t0))
        print(seed, f"tau={p['tau']:.2f}",
              {r["variant"][0] + ":" + r["method"]: round(r["fit"]["tau"], 2)
               for r in rows if r["seed"] == seed}, flush=True)
        json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    a, b = map(int, sys.argv[1].split(":"))
    run(range(a, b), sys.argv[2])
