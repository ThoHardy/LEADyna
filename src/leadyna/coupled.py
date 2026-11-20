"""Coupled LEAD: whole epochs as a chain of 1-D stochastic stages, without time windows.

The multichannel signal is modelled as ``x(t) = sum_k a_k z_k(t) + noise``. Each
``z_k`` is one processing stage: a 1-D latent with LEAD dynamics, driven by the
stimulus (first stage) or by the previous stage after a fitted delay (no feedback).
The method finds the spatial patterns ``a_k``, the number of stages K, and for each
stage its timing and its dynamics (linear, or modulated gain).

Pipeline (:func:`fit_coupled`, or the :class:`CoupledLEAD` wrapper):

1. :func:`find_patterns`: greedy deflation. Take the pattern at the most decodable
   time, test it against label permutations, project it out, repeat.
2. :func:`unmix`: joint GLS estimate of the K latents, in rest-SD units.
3. :func:`fit_cascade`: maximum likelihood (UKF, same scheme as the ``"fast"``
   engine) of every stage on single trials, plus demixing coefficients ``alpha``
   that remove what later-found stages leak into earlier-found latents.
4. :func:`evaluate_K` / :func:`select_K`: cross-validated choice of K.

Conventions: ``X`` is (trials, channels, times); ``cats`` holds one integer per trial,
0 = rest / noise only, 1..L = stimulus levels; stage parameters are in sample units
(``tau``, ``onset``, ``scale``, ``delay``), as in the rest of LEADyna with ``dt = 1``.

Needs JAX (``pip install "leadyna[coupled]"``), used for exact gradients of the UKF
likelihood. Importing this module switches JAX to 64-bit floats.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.stats import rankdata

jax.config.update("jax_enable_x64", True)

__all__ = ["CoupledLEAD", "fit_coupled", "find_patterns", "unmix", "fit_cascade",
           "cascade_loglik", "clean_latents", "demixed_patterns", "evaluate_K", "select_K",
           "simulate_chain", "DEFAULT_BOUNDS"]

SHARPNESS = 5.0   # sigmoid sharpness of the modulated-gain stage, in rest-SD units
MODELS = ("linear", "modulated_gain")
DEFAULT_BOUNDS = dict(
    tau=(1.5, 150.0), process_noise=(0.01, 3.0), measure_noise=(0.01, 3.0),
    onset=(0.0, 50.0), scale=(0.5, 20.0), amp=(0.0, 5.0), coupling=(0.0, 3.0),
    delay=(0.0, 60.0), g0=(-2.0, 2.0), g1=(-3.0, 3.0), g2=(-3.0, 3.0),
    threshold=(0.0, 5.0), alpha=(-3.0, 3.0))


def _pmap(fn, items, n_jobs):
    """Map over threads (JAX releases the GIL, so independent fits run in parallel)."""
    n_jobs = min(n_jobs or os.cpu_count() or 1, len(items))
    if n_jobs <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(n_jobs) as pool:
        return list(pool.map(fn, items))


# =============================================================================
# Steps 1-2: spatial patterns by greedy deflation, joint unmixing
# =============================================================================

def rest_cov(X, cats, ridge=1e-3):
    """Channel covariance of the rest trials (all samples), with a small ridge."""
    S = np.cov(X[cats == 0].transpose(1, 0, 2).reshape(X.shape[1], -1))
    return S + ridge * np.trace(S) / len(S) * np.eye(len(S))


def _decodability(Xs, lab, n_folds=5, reg=0.1, seed=0):
    """Cross-validated AUC of a shrinkage-LDA decoder (lab 1 vs 0) at every sample."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(lab), int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(lab == c))
        folds[idx] = np.arange(len(idx)) % n_folds
    Xt = Xs.transpose(2, 0, 1)                      # (time, trials, channels)
    C = Xt.shape[2]
    aucs = np.zeros((n_folds, len(Xt)))
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        A1, A0 = Xt[:, tr & (lab == 1)], Xt[:, tr & (lab == 0)]
        mu1, mu0 = A1.mean(1), A0.mean(1)
        d1, d0 = A1 - mu1[:, None], A0 - mu0[:, None]
        S = (d1.transpose(0, 2, 1) @ d1 + d0.transpose(0, 2, 1) @ d0) / (tr.sum() - 2)
        shrink = reg * np.trace(S, axis1=1, axis2=2)[:, None, None] / C * np.eye(C)
        w = np.linalg.solve((1 - reg) * S + shrink, (mu1 - mu0)[..., None])
        scores = (Xt[:, te] @ w)[..., 0].T          # (test trials, time)
        n1 = lab[te].sum()
        ranks = rankdata(scores, axis=0)[lab[te] == 1].sum(0)
        aucs[f] = (ranks - n1 * (n1 + 1) / 2) / (n1 * (te.sum() - n1))
    return aucs.mean(0)


def find_patterns(X, cats, times, k_max=4, stim_levels=None, n_perm=20, q_perm=0.95,
                  t_search=(0.0, None), auto_frac=0.8, max_half_win_s=0.15, seed=0):
    """Greedy deflation: most decodable pattern -> project it out -> repeat ``k_max`` times.

    At each step a decoder separates ``stim_levels`` (default: the two strongest)
    from rest at every sample. The pattern ``b`` is the stimulus - rest mean over
    the window around the best time ``t_star`` where the AUC stays above
    ``auto_frac`` of its peak height (at most +-``max_half_win_s``). The AUC peak is
    tested against label permutations (max over time); ``K_perm`` is the number of
    consecutive significant steps, and permutations stop at the first failure.
    ``b`` is then removed by an oblique projection in the rest-noise metric.

    Returns a dict: ``B`` (channels, k_max), ``t_star``, ``window``, ``auc``,
    ``auc_max``, ``thr``, ``significant``, ``K_perm``, ``Sigma`` (rest covariance).
    """
    levels = sorted(set(cats.tolist()) - {0})
    stim_levels = levels[-2:] if stim_levels is None else list(stim_levels)
    Sigma = rest_cov(X, cats)
    Si = np.linalg.inv(Sigma)
    hw_max = int(round(max_half_win_s / (times[1] - times[0])))
    search = (times >= t_search[0]) & (times <= (times[-1] if t_search[1] is None else t_search[1]))
    stim, use = np.isin(cats, stim_levels), np.isin(cats, stim_levels + [0])
    lab = (cats[use] != 0).astype(int)
    rng = np.random.default_rng(seed)
    Xd = X.copy()
    out = dict(B=[], t_star=[], window=[], auc=[], auc_max=[], thr=[], significant=[], Sigma=Sigma)
    testing = True
    for _ in range(k_max):
        Xs = uniform_filter1d(Xd[use], 3, axis=-1, mode="constant")   # 3-sample smoothing
        auc = _decodability(Xs, lab, seed=seed)
        thr = np.nan
        if testing:
            null = [_decodability(Xs, rng.permutation(lab), seed=seed)[search].max()
                    for _ in range(n_perm)]
            thr = float(np.quantile(null, q_perm))
            testing = bool(auc[search].max() > thr)
        ts = np.flatnonzero(search)[np.argmax(auc[search])]
        level = 0.5 + auto_frac * (auc[ts] - 0.5)
        lo = hi = ts
        while lo > 0 and auc[lo - 1] >= level and ts - lo < hw_max:
            lo -= 1
        while hi < len(auc) - 1 and auc[hi + 1] >= level and hi - ts < hw_max:
            hi += 1
        win = slice(max(min(lo, ts - 1), 0), max(hi, ts + 1) + 1)
        b = Xd[stim][:, :, win].mean((0, 2)) - Xd[cats == 0][:, :, win].mean((0, 2))
        out["B"].append(b)
        out["t_star"].append(float(times[ts]))
        out["window"].append((float(times[win][0]), float(times[win][-1])))
        out["auc"].append(auc)
        out["auc_max"].append(float(auc[search].max()))
        out["thr"].append(thr)
        out["significant"].append(testing)
        Xd = np.einsum("cd,ndt->nct", np.eye(len(b)) - np.outer(b, b @ Si) / (b @ Si @ b), Xd)
    out["B"] = np.stack(out["B"], 1)
    out["K_perm"] = int(sum(out["significant"]))
    return out


def unmix(X, B, Sigma):
    """GLS latents ``(B' S^-1 B)^-1 B' S^-1 x``: (trials, components, times)."""
    Si = np.linalg.inv(Sigma)
    return np.einsum("kc,nct->nkt", np.linalg.solve(B.T @ Si @ B, B.T @ Si), X)


def _scaled_latents(X, cats, pat, scale=None):
    """Unmixed latents in rest-SD units, and the scale used."""
    Z = unmix(X, pat["B"], pat["Sigma"])
    scale = Z[cats == 0].std(axis=(0, 2)) if scale is None else scale
    return Z / scale[None, :, None], scale


# =============================================================================
# Step 3a: one stage = 1-D stochastic node, UKF likelihood in JAX
# =============================================================================

def _gamma_bump(t, onset, scale):
    """Stimulus input kernel: 0 before ``onset``, peak 1 at ``onset + 2 * scale``."""
    x = (t - onset) / scale
    return jnp.where(x > 0, (jnp.maximum(x, 1e-12) ** 2 / 4.0) * jnp.exp(2.0 - x), 0.0)


def _shift(u, delay):
    """Delay each row of ``u`` by a continuous number of samples (linear interpolation)."""
    T = u.shape[1]
    idx = jnp.arange(T) - delay
    i0 = jnp.floor(idx)
    frac = idx - i0
    i0 = i0.astype(int)
    take = lambda i: jnp.where((i >= 0) & (i < T), u[:, jnp.clip(i, 0, T - 1)], 0.0)
    return (1 - frac) * take(i0) + frac * take(i0 + 1)


def _ukf(y, u, tau, w, q, r, g0, g1, g2, thr):
    """UKF over time, vectorized over trials (Julier sigma points, kappa = 0: x +- sqrt(P)).

    ``z[t+1] = z[t] - z[t]/tau + w*u[t] + gain(u[t]) * sigmoid(5*(z[t] - thr)) + q*eps``,
    ``gain(u) = relu(g0 + g1*u + g2*u^2)``, ``y[t] = z[t] + r*eta``. Every trial starts
    from the stationary distribution of the linear part. Returns per-sample
    log-likelihoods and the moments needed for smoothing.
    """
    a = 1.0 - 1.0 / tau
    n = y.shape[0]

    def fx(x, ut):
        gain = jax.nn.relu(g0 + g1 * ut + g2 * ut ** 2)
        return x + (-x / tau + w * ut + gain * jax.nn.sigmoid(SHARPNESS * (x - thr)))

    def step(carry, obs):
        m, P = carry
        yt, ut = obs
        S = P + r ** 2
        e = yt - m
        ll = -0.5 * (jnp.log(2 * jnp.pi * S) + e ** 2 / S)
        mf, Pf = m + P / S * e, (1 - P / S) * P                  # update
        sd = jnp.sqrt(Pf)
        fp, fm = fx(mf + sd, ut), fx(mf - sd, ut)                # predict
        mp = 0.5 * (fp + fm)
        Pp = 0.5 * ((fp - mp) ** 2 + (fm - mp) ** 2) + q ** 2
        return (mp, Pp), (ll, mf, mp, Pp, 0.5 * sd * (fp - fm))

    init = (jnp.zeros(n), jnp.full(n, q ** 2 / jnp.maximum(1 - a ** 2, 1e-3)))
    return lax.scan(step, init, (y.T, u.T))[1]


class _Stage:
    """Parameter layout of one stage.

    ``first``: driven by the stimulus (``onset``, ``scale``, one ``amp`` per level)
    rather than by its parent stage (``coupling``, ``delay``). ``gain``: modulated-gain
    dynamics (``g0, g1, g2, threshold``). ``later``: indices of the later-found
    components whose smoothed latents are subtracted with weights ``alpha{j}``.
    """

    def __init__(self, first, gain, n_levels=0, later=()):
        self.first, self.gain, self.n_levels, self.later = first, gain, n_levels, tuple(later)
        names = ["tau", "process_noise", "measure_noise"]
        names += (["onset", "scale"] + [f"amp{i + 1}" for i in range(n_levels)] if first
                  else ["coupling", "delay"])
        names += ["g0", "g1", "g2", "threshold"] if gain else []
        self.names = names + [f"alpha{j}" for j in self.later]
        self.model = MODELS[gain]

    def bounds(self, custom=None):
        B = {**DEFAULT_BOUNDS, **(custom or {})}
        return [B[n] if n in B else B[n.rstrip("0123456789")] for n in self.names]

    def run(self, theta, data):
        """Filter the stage's data: ``y`` (trials, times), ``others`` (len(later), trials,
        times), and ``cat`` + ``t0`` (first stage) or ``parent`` (downstream stage)."""
        p = dict(zip(self.names, theta))
        if self.first:
            h = _gamma_bump(jnp.arange(data["y"].shape[1]) - data["t0"], p["onset"], p["scale"])
            amps = jnp.stack([0.0 * theta[0]] + [p[f"amp{i + 1}"] for i in range(self.n_levels)])
            u, w = amps[data["cat"]][:, None] * h[None, :], 1.0
        else:
            u, w = _shift(data["parent"], p["delay"]), p["coupling"]
        y = data["y"]
        for a, j in enumerate(self.later):
            y = y - p[f"alpha{j}"] * data["others"][a]
        g = [p[k] for k in ("g0", "g1", "g2", "threshold")] if self.gain else [-1.0, 0.0, 0.0, 0.0]
        return _ukf(y, u, p["tau"], w, p["process_noise"], p["measure_noise"], *g)


_JIT = {}


def _nll(stage):
    """Jitted (value-and-gradient, value) of the negative log-likelihood, cached by layout."""
    key = (stage.first, stage.gain, stage.n_levels, len(stage.later))
    if key not in _JIT:
        f = lambda theta, data: -jnp.sum(stage.run(theta, data)[0])
        _JIT[key] = (jax.jit(jax.value_and_grad(f)), jax.jit(f))
    return _JIT[key]


def _to_jax(data):
    return {k: jnp.asarray(v) for k, v in data.items()}


def _fit(stage, data, inits, bounds=None, maxiter=400, n_jobs=None):
    """Bounded L-BFGS-B from every init (in parallel threads); keep the best."""
    vg, jd = _nll(stage)[0], _to_jax(data)
    bnds = stage.bounds(bounds)
    lo, hi = np.array(bnds).T

    def run(x0):
        fun = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), jd))
        try:
            res = minimize(fun, np.clip(np.asarray(x0, float), lo, hi), jac=True,
                           method="L-BFGS-B", bounds=bnds, options=dict(maxiter=maxiter))
        except Exception:
            return None
        return res if np.isfinite(res.fun) else None

    results = [r for r in _pmap(run, inits, n_jobs) if r is not None]
    if not results:
        raise RuntimeError("Stage fit failed from every starting point.")
    best = min(results, key=lambda r: r.fun)
    return dict(stage=stage, model=stage.model, theta=best.x, nll=float(best.fun),
                params=dict(zip(stage.names, best.x.tolist())))


def _loglik(fit, data):
    return -float(_nll(fit["stage"])[1](jnp.asarray(fit["theta"]), _to_jax(data)))


def _smooth(fit, data):
    """Unscented RTS smoother: denoised latent of a fitted stage, (trials, times)."""
    outs = fit["stage"].run(jnp.asarray(fit["theta"]), _to_jax(data))
    _, mf, mp, Pp, C = (np.asarray(o) for o in outs)
    ms = mf.copy()
    for t in range(len(mf) - 2, -1, -1):
        ms[t] = mf[t] + C[t] / Pp[t] * (ms[t + 1] - mp[t])
    return ms.T


def _gain_inits(linear_fit, gain_stage):
    """Starting points of the modulated-gain fit, built around the linear solution."""
    base = dict(linear_fit["params"])
    return [[{**base, "tau": tau, "g0": g0, "g1": g1, "threshold": thr}.get(n, 0.0)
             for n in gain_stage.names]
            for tau in (base["tau"], 6.0, 15.0)
            for g0, g1, thr in ((0.0, 0.2, 1.5), (0.3, 0.0, 2.0), (0.6, 0.0, 3.0))]


def _fit_models(data, first, n_levels, later, dt, models, bounds, delay_grid_s, warm, n_jobs):
    """Fit the linear and (if requested) modulated-gain versions of one stage.

    Cold start: the linear fit is multi-start over the input onset (first stage) or
    profiled over a grid of delays (downstream stage); the gain fit starts from it.
    Warm start (``warm`` = fits of the previous pass): restart from them.
    """
    lin, mg = _Stage(first, False, n_levels, later), _Stage(first, True, n_levels, later)
    fits = {}
    if warm is not None:
        for stage in (lin, mg):
            if stage.model not in warm:
                continue
            prev = warm[stage.model]
            n_old = len(prev["stage"].later)
            base = list(prev["theta"][:len(prev["theta"]) - n_old])
            alpha = list(prev["theta"][len(base):]) if n_old == len(later) else [0.0] * len(later)
            inits = [base + alpha] + ([base + [0.0] * len(later)] if any(alpha) else [])
            if stage.gain and "linear" in fits:
                inits += _gain_inits(fits["linear"], mg)[1:]
            fits[stage.model] = _fit(stage, data, inits, bounds, n_jobs=n_jobs)
        return fits
    zeros = [0.0] * len(later)
    if first:
        inits = [[5.0, 0.3, 0.5, onset / dt, scale / dt] + [0.5] * n_levels + zeros
                 for onset in (0.02, 0.08, 0.16, 0.3) for scale in (0.02, 0.06)]
        fits["linear"] = _fit(lin, data, inits, bounds, n_jobs=n_jobs)
    else:
        grid = np.arange(0, 0.42, 0.04) if delay_grid_s is None else delay_grid_s

        def at_delay(d):   # delay held fixed at one grid value
            b = {**(bounds or {}), "delay": (d / dt, d / dt + 1e-9)}
            return _fit(lin, data, [[8.0, 0.3, 0.5, 0.2, d / dt] + zeros], b, maxiter=200, n_jobs=1)

        best = min(_pmap(at_delay, list(grid), n_jobs), key=lambda f: f["nll"])
        fits["linear"] = _fit(lin, data, [best["theta"]], bounds)
    if "modulated_gain" in models:
        fits["modulated_gain"] = _fit(mg, data, _gain_inits(fits["linear"], mg), bounds,
                                      n_jobs=n_jobs)
    return fits


# =============================================================================
# Step 3b: time-ordered chain with demixing
# =============================================================================

def _stage_data(comp, Zs, cats, t0, smooths, use_alpha=True):
    """Data of one stage: its latent, its parent's smoothed latent, the latents to demix."""
    y = Zs[:, comp["idx"]]
    later = comp["later"] if use_alpha else []
    others = np.stack([smooths[j] for j in later]) if later else np.zeros((0,) + y.shape)
    data = dict(y=y, others=others)
    if comp["parent"] is None:
        data.update(cat=cats, t0=t0)
    else:
        data["parent"] = smooths[comp["parent"]]
    return data


def fit_cascade(Zs, cats, times, t_star, K, models=MODELS, bounds=None, select="bic",
                delay_grid_s=None, n_iter=3, n_jobs=None, verbose=False, _cache=None):
    """Fit the chain of the first ``K`` components of ``Zs`` (trials, components, times).

    Components are ordered by ``t_star`` (time order = processing order): the earliest
    is driven by the stimulus, each other one by the smoothed latent of the previous
    one. Pass 1 fits the chain as is. Passes 2..``n_iter`` refit it with demixing:
    component ``i`` is fitted on ``Zs[:, i] - sum_j alpha_ij * smoothed_j``, ``j`` over
    the components found after ``i`` in the deflation. Each stage keeps the model with
    the lowest BIC (``select="bic"``) or the highest likelihood (``"ll"``).

    Returns a dict: ``order`` (component indices in time order), ``comps`` (per
    component: ``fits`` by model, ``crit``, ``chosen``, ``smooth``, ``parent``, ``later``).
    """
    dt = float(times[1] - times[0])
    t0 = int(np.searchsorted(times, -1e-9))
    n_levels = int(cats.max())
    order = sorted(range(K), key=lambda i: t_star[i])
    comps = {i: dict(idx=i, parent=order[p - 1] if p else None, later=list(range(i + 1, K)))
             for p, i in enumerate(order)}
    smooths = {i: np.zeros_like(Zs[:, i]) for i in range(K)}
    for it in range(n_iter if K > 1 else 1):
        for p, i in enumerate(order):
            c = comps[i]
            key = tuple(order[:p + 1])     # pass 1 only depends on the chain up to this stage
            if it == 0 and _cache is not None and key in _cache:
                c.update(_cache[key])
            else:
                data = _stage_data(c, Zs, cats, t0, smooths, use_alpha=it > 0)
                first = c["parent"] is None
                fits = _fit_models(data, first, n_levels if first else 0, c["later"] if it else [],
                                   dt, models, bounds, delay_grid_s, c.get("fits"), n_jobs)
                penalty = np.log(data["y"].size) if select == "bic" else 0.0
                crit = {m: 2 * f["nll"] + len(f["theta"]) * penalty for m, f in fits.items()}
                chosen = min(crit, key=crit.get)
                c.update(fits=fits, crit=crit, chosen=chosen, smooth=_smooth(fits[chosen], data))
                if it == 0 and _cache is not None:
                    _cache[key] = {k: c[k] for k in ("fits", "crit", "chosen", "smooth")}
            smooths[i] = c["smooth"]
        if verbose:
            chosen = {i: comps[i]["fits"][comps[i]["chosen"]] for i in order}
            print(f"pass {it + 1}: " + " | ".join(
                f"c{i} [{f['model']}] " + " ".join(f"{k}={v:.3g}" for k, v in f["params"].items())
                for i, f in chosen.items()))
    return dict(order=order, comps=comps, K=K, t0=t0, dt=dt)


def cascade_loglik(casc, Zs, cats, n_pass=3):
    """Log-likelihood of new latents under a fitted chain (parameters fixed).

    Smoothed latents are recomputed on the new data, iterating as in the fit.
    Returns (LL of the chosen model per component, LL of every model, smoothed latents).
    """
    K = casc["K"]
    smooths = {i: np.zeros_like(Zs[:, i]) for i in range(K)}
    for p in range(n_pass if K > 1 else 1):
        per_model = {}
        for i in casc["order"]:
            c = casc["comps"][i]
            data = _stage_data(c, Zs, cats, casc["t0"], smooths)
            if p == 0:
                data["others"] = np.zeros_like(data["others"])
            per_model[i] = {m: _loglik(f, data) for m, f in c["fits"].items()}
            smooths[i] = _smooth(c["fits"][c["chosen"]], data)
    return {i: per_model[i][casc["comps"][i]["chosen"]] for i in per_model}, per_model, smooths


def clean_latents(casc, Zs, cats):
    """Smoothed, demixed latents of a fitted chain: (trials, K, times)."""
    smooths = cascade_loglik(casc, Zs, cats)[2]
    return np.stack([smooths[i] for i in range(casc["K"])], 1)


def demixed_patterns(casc, B_scaled):
    """Full topographies ``A = B_scaled @ M``, M unit upper-triangular holding the alphas."""
    K = casc["K"]
    M = np.eye(K)
    for i, c in casc["comps"].items():
        for j in c["later"]:
            M[i, j] = c["fits"][c["chosen"]]["params"].get(f"alpha{j}", 0.0)
    return B_scaled[:, :K] @ M


# =============================================================================
# Step 4 and full pipeline
# =============================================================================

def _null_fit(y):
    """Input-free OU process: the model of a direction that holds no stage."""
    stage = _Stage(False, False)
    fixed = dict(coupling=(0.0, 1e-9), delay=(0.0, 1e-9))
    return _fit(stage, dict(y=y, parent=np.zeros_like(y)),
                [[5.0, 0.3, 0.5, 0.0, 0.0], [20.0, 0.1, 0.8, 0.0, 0.0]], fixed)


def evaluate_K(X, cats, times, Ks=(1, 2, 3), k_max=4, n_folds=3, models=MODELS, bounds=None,
               stim_levels=None, n_perm=20, seed=0, n_jobs=None, verbose=True):
    """Cross-validate the number of stages.

    In each fold (trials stratified by category), patterns, noise covariance, scales
    and fits come from the training trials only. The score is the held-out
    log-likelihood of the ``k_max`` unmixed latents: the first K under the fitted
    chain, the remaining ones under a null OU process, so that all K (and K = 0)
    are scored on the same data. Feed the result to :func:`select_K`.
    """
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(cats), int)
    for c in np.unique(cats):
        idx = rng.permutation(np.flatnonzero(cats == c))
        folds[idx] = np.arange(len(idx)) % n_folds
    res = dict(Ks=tuple(Ks), null_ll=[], K_perm=[], ll={K: [] for K in Ks})
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        pat = find_patterns(X[tr], cats[tr], times, k_max, stim_levels, n_perm, seed=seed)
        Ztr, scale = _scaled_latents(X[tr], cats[tr], pat)
        Zte, _ = _scaled_latents(X[te], cats[te], pat, scale)
        null = []
        for j in range(k_max):
            fit = _null_fit(Ztr[:, j])
            null.append(_loglik(fit, dict(y=Zte[:, j], parent=np.zeros_like(Zte[:, j]))))
        res["null_ll"].append(sum(null))
        res["K_perm"].append(pat["K_perm"])
        cache = {}
        for K in Ks:
            casc = fit_cascade(Ztr, cats[tr], times, pat["t_star"], K, models, bounds,
                               n_jobs=n_jobs, _cache=cache)
            ll = sum(cascade_loglik(casc, Zte, cats[te])[0].values()) + sum(null[K:])
            res["ll"][K].append(ll)
            if verbose:
                print(f"fold {f + 1}/{n_folds}  K={K}  held-out LL = {ll:.1f}  "
                      f"(null {sum(null):.1f})")
    return res


def select_K(res, elbow_frac=0.1):
    """Number of stages according to each rule, from the output of :func:`evaluate_K`.

    ``K_perm`` (recommended): median over folds of the number of consecutive
    significant deflation steps. ``K_elbow`` (check): 0 unless K = 1 beats the null
    in every fold; then stages are added while the next one gains at least
    ``elbow_frac`` of the first stage's held-out gain. ``K_ll`` (max held-out LL)
    and ``K_ll_1se`` (smallest K within 1 SE of the best) over-select: the
    demixing can exploit noise correlations between latent directions.
    """
    Ks = (0,) + tuple(res["Ks"])
    per_fold = np.array([res["null_ll"]] + [res["ll"][K] for K in res["Ks"]]).T   # (folds, Ks)
    ll = per_fold.sum(0)
    best = int(np.argmax(ll))
    diffs = per_fold - per_fold[:, [best]]
    se = diffs.std(0, ddof=1) * np.sqrt(len(diffs)) if len(diffs) > 1 else np.zeros(len(Ks))
    gains = np.diff(ll)
    K_elbow = 0
    if gains[0] > 0 and (per_fold[:, 1] > per_fold[:, 0]).all():
        K_elbow = Ks[1]
        for i in range(1, len(gains)):
            if gains[i] / gains[0] < elbow_frac:
                break
            K_elbow = Ks[i + 1]
    return dict(K_perm=int(np.median(res["K_perm"])), K_elbow=K_elbow, K_ll=Ks[best],
                K_ll_1se=next(k for k, v, s in zip(Ks, ll, se) if v >= ll[best] - s),
                ll=dict(zip(Ks, ll.tolist())))


def fit_coupled(X, cats, times, K=None, k_max=4, models=MODELS, bounds=None, stim_levels=None,
                n_perm=20, seed=0, n_jobs=None, verbose=False):
    """Full pipeline on all trials: patterns -> unmixing -> chain of ``K`` stages.

    ``K=None`` uses ``K_perm``, the permutation stop rule of :func:`find_patterns`.
    Returns a dict: ``K``, ``patterns`` (output of :func:`find_patterns`), ``scale``
    and ``Zs`` (raw latents in rest-SD units), ``cascade`` (output of
    :func:`fit_cascade`), ``A`` (demixed topographies, channels x K) and ``latents``
    (smoothed, demixed latents, trials x K x times). Columns follow the order of
    discovery; ``cascade["order"]`` gives the time order.
    """
    pat = find_patterns(X, cats, times, max(k_max, K or 0), stim_levels, n_perm, seed=seed)
    K = pat["K_perm"] if K is None else K
    if K == 0:
        raise ValueError("No significant stimulus-related component (K_perm = 0). "
                         "Pass K explicitly to fit a chain anyway.")
    Zs, scale = _scaled_latents(X, cats, pat)
    casc = fit_cascade(Zs, cats, times, pat["t_star"], K, models, bounds, n_jobs=n_jobs,
                       verbose=verbose)
    return dict(K=K, patterns=pat, scale=scale, Zs=Zs, cascade=casc,
                A=demixed_patterns(casc, pat["B"] * scale),
                latents=np.stack([casc["comps"][i]["smooth"] for i in range(K)], 1))


class CoupledLEAD:
    """Chain of coupled 1-D LEAD stages fitted on whole epochs (see the module docstring).

    >>> model = CoupledLEAD().fit(X, cats, times)      # K from the permutation rule
    >>> model.summary()                                # stages in time order
    >>> model.A_, model.latents_                       # topographies, smoothed latents
    >>> model.select_K(X, cats, times)                 # cross-validated check of K
    >>> model.export_html("results.html", info)        # interactive 3-D viewer

    ``options`` are passed to :func:`fit_coupled` and :func:`evaluate_K`
    (``k_max``, ``models``, ``bounds``, ``stim_levels``, ``n_perm``, ``seed``, ``n_jobs``).
    """

    def __init__(self, K=None, **options):
        self.K, self.options = K, options

    def fit(self, X, cats, times, verbose=False):
        self.result_ = fit_coupled(X, cats, times, self.K, verbose=verbose, **self.options)
        self.K_, self.A_, self.latents_ = (self.result_[k] for k in ("K", "A", "latents"))
        return self

    def select_K(self, X, cats, times, Ks=(1, 2, 3), n_folds=3, verbose=True):
        """Cross-validate K; returns the choice of each rule (see :func:`select_K`)."""
        res = evaluate_K(X, cats, times, Ks, n_folds=n_folds, verbose=verbose, **self.options)
        return select_K(res)

    def export_html(self, path, info=None, **kwargs):
        """Interactive 3-D viewer of this fit, as one HTML file (see
        :func:`leadyna.coupled_viewer.export_html` for the options)."""
        from .coupled_viewer import export_html
        return export_html(self, path, info, **kwargs)

    def summary(self):
        """One dict per stage, in time order, with times converted to seconds."""
        casc, out = self.result_["cascade"], []
        for i in casc["order"]:
            c = casc["comps"][i]
            p = c["fits"][c["chosen"]]["params"]
            row = dict(component=i, model=c["chosen"], parent=c["parent"],
                       t_star_s=self.result_["patterns"]["t_star"][i])
            in_samples = ("tau", "onset", "scale", "delay")
            row.update({f"{k}_s": p[k] * casc["dt"] for k in in_samples if k in p})
            row.update({k: v for k, v in p.items() if k not in in_samples})
            row["delta_bic"] = {m: float(v - c["crit"][c["chosen"]]) for m, v in c["crit"].items()}
            out.append(row)
        return out


# =============================================================================
# Ground-truth simulator (tests, parameter recovery)
# =============================================================================

def simulate_chain(stages, pattern_corr, n_channels=32, n_per_level=100,
                   levels=(0.25, 0.5, 0.75, 1.0), tmin=-0.5, tmax=1.0, dt=0.01, pattern_amp=0.8,
                   sensor_noise=1.0, n_background=5, background_rho=0.9, seed=0):
    """Simulate epochs from a chain of stages mixed into channels.

    ``stages``: one dict per stage, in time order, with ``tau`` (samples) and
    ``process_noise``; ``onset``, ``scale`` (seconds) and ``amp`` for the first stage,
    ``coupling`` and ``delay`` (seconds) for the others; optional ``gain=(g0, g1, g2)``
    and ``threshold`` for modulated-gain dynamics. ``pattern_corr``: K x K correlation
    matrix of the topographies. Noise: ``n_background`` slow AR(1) sources + white
    sensor noise. Rest trials (category 0) come first, then ``n_per_level`` per level.
    Returns a dict with ``X``, ``cats``, ``times`` and the ground truth ``Z``, ``A``.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(tmin, tmax, dt)
    T, K = len(times), len(stages)
    strength = np.r_[0.0, levels]
    cats = np.repeat(np.arange(len(strength)), n_per_level)
    n = len(cats)
    Q, _ = np.linalg.qr(rng.normal(size=(n_channels, K)))
    A = Q @ np.linalg.cholesky(np.asarray(pattern_corr, float)).T * pattern_amp
    Z = np.zeros((n, K, T))
    for k, s in enumerate(stages):
        if k == 0:
            x = (times - s["onset"]) / s["scale"]
            bump = np.where(x > 0, np.maximum(x, 1e-12) ** 2 / 4.0 * np.exp(2.0 - x), 0.0)
            u = s["amp"] * strength[cats][:, None] * bump[None, :]
        else:
            d = int(round(s["delay"] / dt))
            u = np.zeros((n, T))
            u[:, d:] = s["coupling"] * Z[:, k - 1, :T - d]
        g, thr = s.get("gain"), s.get("threshold", 0.0)
        z = Z[:, k]
        for t in range(T - 1):
            gain = np.maximum(0.0, g[0] + g[1] * u[:, t] + g[2] * u[:, t] ** 2) if g else 0.0
            feedback = gain / (1 + np.exp(-SHARPNESS * (z[:, t] - thr)))
            drift = -z[:, t] / s["tau"] + u[:, t] + feedback
            z[:, t + 1] = z[:, t] + drift + s["process_noise"] * rng.normal(size=n)
    mixing = rng.normal(size=(n_channels, n_background))
    bg = np.zeros((n, n_background, T))
    bg[:, :, 0] = rng.normal(size=(n, n_background))
    for t in range(T - 1):
        innovation = np.sqrt(1 - background_rho ** 2) * rng.normal(size=(n, n_background))
        bg[:, :, t + 1] = background_rho * bg[:, :, t] + innovation
    X = (np.einsum("ck,nkt->nct", A, Z) + np.einsum("cb,nbt->nct", mixing, bg)
         + sensor_noise * rng.normal(size=(n, n_channels, T)))
    return dict(X=X, cats=cats, times=times, Z=Z, A=A)
