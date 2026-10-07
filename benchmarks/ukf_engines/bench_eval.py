"""Accuracy + cost of one log-likelihood evaluation, realistic infant-sized data (see README.md)."""
import time

import numpy as np
from engines import _cat_ll_grad, _cat_ll_jit, ll_numba, ll_numpy

from leadyna import LatentSeries
from leadyna.model import LinearLEAD, StratifiedGainModulationLEAD


def make_data(n_trials=150, seed=0):
    np.random.seed(seed)
    truth = StratifiedGainModulationLEAD(
        tau=10.0, process_noise=0.3, measure_noise=0.3, threshold=1.0, sharpness=5,
        n_categories=5, w1=0.02, w2=0.04, w3=0.06, w4=0.08, g1=0.05, g2=0.1, g3=0.15, g4=0.2)
    box = np.r_[np.zeros(60), np.ones(40), np.zeros(20)]          # -0.2..1.0 s at 100 Hz
    inputs = {c: np.tile(box * (c > 0), (n_trials, 1)) for c in range(5)}
    sim = truth.measure_simulations(inputs)
    seg = 20
    base = [sim[0][:, k * seg:(k + 1) * seg] for k in range(120 // seg)]
    base += [sim[c][:, :seg] for c in range(1, 5)]
    states = {c: sim[c][:, 60:100] for c in range(1, 5)}
    states[0] = np.concatenate(base)
    ins = {c: np.ones_like(states[c]) for c in range(1, 5)}
    ins[0] = np.zeros_like(states[0])
    return truth, LatentSeries(states, ins)


def gm_vec_factory(m):
    g = np.array([getattr(m, f"g{i}") for i in range(m.n_categories)])

    def factory(cat):
        w = getattr(m, f"w{cat}")

        def fx(x, dt, u):
            idx = (cat * u).astype(int)
            sig = 1 / (1 + np.exp(m.sharpness * (m.threshold - x)))
            return x + (-x / m.tau + w * u + g[idx] * sig) * m.dt
        return fx
    return factory


def gm_numba_params(m):
    def params(cat):
        return dict(tau=m.tau, w=getattr(m, f"w{cat}"), a=0.0, b=0.0, g_off=m.g0,
                    g_on=getattr(m, f"g{cat}"), th=m.threshold, sh=m.sharpness, use_sig=True)
    return params


def gm_jax(m, data):
    tot = 0.0
    for cat in data.categories:
        prm = (m.tau, getattr(m, f"w{cat}"), 0.0, 0.0, m.g0, getattr(m, f"g{cat}"),
               m.threshold, float(m.sharpness))
        tot += _cat_ll_jit(data.states[cat], data.inputs[cat], 1.0, m.process_noise,
                           m.measure_noise, prm, cat)
    return float(tot)


def timeit(fn, n=3):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    return out, (time.perf_counter() - t) / n


if __name__ == "__main__":
    truth, data = make_data()
    n_samples = sum(x.size for x in data.states.values())
    print(f"data: {n_samples} samples, "
          f"{sum(x.shape[0] for x in data.states.values())} sequences")

    m = truth
    t = time.perf_counter()
    ref = m.loglikelihood(data, n_jobs=1, engine="filterpy")
    t_ref = time.perf_counter() - t
    print(f"filterpy (current)     ll={ref:.10f}  {t_ref*1e3:9.1f} ms")

    for name, fn in [
        ("numpy vectorized", lambda: ll_numpy(data.states, data.inputs, 1.0, m.process_noise,
                                               m.measure_noise, gm_vec_factory(m))),
        ("numba kernel", lambda: ll_numba(data.states, data.inputs, 1.0, m.process_noise,
                                          m.measure_noise, gm_numba_params(m))),
        ("jax scan (jit)", lambda: gm_jax(m, data)),
    ]:
        val, dt_ = timeit(fn)
        print(f"{name:22s} ll={val:.10f}  {dt_*1e3:9.2f} ms   |diff|={abs(val-ref):.2e}  "
              f"speed-up x{t_ref/dt_:,.0f}")

    lin = LinearLEAD(tau=10, process_noise=0.3, measure_noise=0.3, n_categories=5,
                     w1=0.02, w2=0.04, w3=0.06, w4=0.08)
    ref_l = lin.loglikelihood(data, n_jobs=1, engine="filterpy")
    val = ll_numpy(data.states, data.inputs, 1.0, 0.3, 0.3, lin._make_fx)
    print(f"linear model: numpy reuses the model's own _make_fx unchanged, "
          f"|diff|={abs(val-ref_l):.2e}, exact Kalman |diff|="
          f"{abs(lin.loglikelihood_kalman(data)-ref_l):.2e}")

    # value + gradient in one pass (JAX), vs finite differences
    cat = 4
    prm = tuple(float(v) for v in (m.tau, m.w4, 0.0, 0.0, m.g0, m.g4, m.threshold, m.sharpness))
    y, u = data.states[cat], data.inputs[cat]
    _cat_ll_grad(y, u, 1.0, 0.3, 0.3, prm, cat)
    t = time.perf_counter()
    for _ in range(5):
        v, g = _cat_ll_grad(y, u, 1.0, 0.3, 0.3, prm, cat)
    t_g = (time.perf_counter() - t) / 5
    _cat_ll_jit(y, u, 1.0, 0.3, 0.3, prm, cat)
    t = time.perf_counter()
    for _ in range(5):
        _cat_ll_jit(y, u, 1.0, 0.3, 0.3, prm, cat)
    t_v = (time.perf_counter() - t) / 5
    print(f"jax one category: value {t_v*1e3:.2f} ms, value+full gradient {t_g*1e3:.2f} ms "
          f"(finite differences over 10 params would cost ~11 values)")
