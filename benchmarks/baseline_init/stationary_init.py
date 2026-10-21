import json
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
from recovery_chunks import METHODS, baseline_correct, build, draw_params, simulate


def kalman_ll(y, tau, pn, mn, init):
    a, Q, R = 1 - 1 / tau, pn ** 2, mn ** 2
    if init == "stationary":
        x, P, t0 = np.zeros(y.shape[0]), np.full(y.shape[0], Q / (1 - a ** 2)), 0
    else:
        x, P, t0 = y[:, 0].copy(), np.ones(y.shape[0]), 1
    ll = 0.0
    for t in range(t0, y.shape[1]):
        if t > 0:
            x, P = a * x, a * a * P + Q
        S = P + R
        e = y[:, t] - x
        ll -= 0.5 * np.sum(np.log(2 * np.pi * S) + e * e / S)
        K = P / S
        x, P = x + K * e, (1 - K) * P
    return ll


def fit_rest(y, init):
    best = None
    for tau0 in (3, 10):
        r = minimize(lambda p: -kalman_ll(y, *p, init), [tau0, .4, .4], method="L-BFGS-B",
                     bounds=[(1.05, 25), (.01, 1), (.01, 1)])
        if best is None or r.fun < best.fun:
            best = r
    return best.x


if __name__ == "__main__":
    a, b = map(int, sys.argv[1].split(":"))
    rows = []
    for seed in range(a, b):
        p = draw_params(np.random.default_rng(1000 + seed))
        sim0 = simulate(p, seed)
        for variant in ("raw", "bc"):
            sim = baseline_correct(sim0) if variant == "bc" else sim0
            for method in METHODS:
                y = build(sim, method).states[0]
                for init in ("leadyna", "stationary"):
                    tau, pn, mn = fit_rest(y, init)
                    rows.append(dict(seed=seed, variant=variant, method=method, init=init,
                                     truth=p, fit=dict(tau=tau, process_noise=pn, measure_noise=mn)))
        json.dump(rows, open(sys.argv[2], "w"), default=float)
        print(seed, flush=True)
