"""Candidate fast UKF log-likelihood engines (1-D state, Julier kappa=0, as in LEADyna).

With n=1 and kappa=0 the sigma set is {x, x+sqrt(P), x-sqrt(P)} with weights {0, 1/2, 1/2},
so one UKF step = two evaluations of f. All engines below implement exactly the
filterpy recursion used by leadyna.model.compute_ukf_loglikelihood.
"""
import numpy as np

LOG2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------------------
# Option A: numpy, vectorized over trials (loop over time only)
# ---------------------------------------------------------------------------
def ll_numpy(state_series, input_series, dt, q, r, fx_factory):
    Q, R = q**2 * dt, r**2
    total = 0.0
    for cat in input_series:
        y, u = state_series[cat], input_series[cat]
        fx = fx_factory(cat)
        x = y[:, 0].copy()
        P = np.ones(len(x))
        ll = 0.0
        for t in range(1, y.shape[1]):
            s = np.sqrt(P)
            f1, f2 = fx(x + s, dt, u[:, t - 1]), fx(x - s, dt, u[:, t - 1])
            xp = 0.5 * (f1 + f2)
            Pp = 0.5 * ((f1 - xp) ** 2 + (f2 - xp) ** 2) + Q
            S = Pp + R
            e = y[:, t] - xp
            ll += -0.5 * np.sum(LOG2PI + np.log(S) + e**2 / S)
            K = Pp / S
            x = xp + K * e
            P = Pp - K * K * S
        total += ll
    return float(total)


# ---------------------------------------------------------------------------
# Option B: numba, compiled scalar loops, one kernel for the whole model family
#   f(x,u) = x + dt*(-x/tau + w*u + (a*x + b + g(u)) * sigmoid(sh*(x - th)))
#   g(u) = g_on if int(cat*u) != 0 else g_off   (gain-modulation semantics)
# ---------------------------------------------------------------------------
try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _kernel(y, u, dt, Q, R, tau, w, a, b, g_off, g_on, th, sh, cat, use_sig):
        n, T = y.shape
        tot = 0.0
        for i in range(n):
            x = y[i, 0]
            P = 1.0
            for t in range(1, T):
                ut = u[i, t - 1]
                s = np.sqrt(P)
                fs = np.empty(2)
                for k in range(2):
                    xs = x + s if k == 0 else x - s
                    d = -xs / tau + w * ut
                    if use_sig:
                        g = g_on if int(cat * ut) != 0 else g_off
                        d += (a * xs + b + g) / (1.0 + np.exp(sh * (th - xs)))
                    fs[k] = xs + d * dt
                xp = 0.5 * (fs[0] + fs[1])
                Pp = 0.5 * ((fs[0] - xp) ** 2 + (fs[1] - xp) ** 2) + Q
                S = Pp + R
                e = y[i, t] - xp
                tot += -0.5 * (LOG2PI + np.log(S) + e * e / S)
                K = Pp / S
                x = xp + K * e
                P = Pp - K * K * S
        return tot

    def ll_numba(state_series, input_series, dt, q, r, params_for_cat):
        Q, R = q**2 * dt, r**2
        total = 0.0
        for cat in input_series:
            p = params_for_cat(cat)
            total += _kernel(state_series[cat], input_series[cat], dt, Q, R, p["tau"], p["w"],
                             p["a"], p["b"], p["g_off"], p["g_on"], p["th"], p["sh"], cat,
                             p["use_sig"])
        return float(total)
except ImportError:  # pragma: no cover
    ll_numba = None


# ---------------------------------------------------------------------------
# Option C: JAX, lax.scan over time, vmap over trials, autodiff gradient
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def _f(x, u, dt, tau, w, a, b, g_off, g_on, th, sh, cat):
        g = jnp.where(jnp.floor(cat * u) != 0, g_on, g_off)
        return x + dt * (-x / tau + w * u + (a * x + b + g) / (1 + jnp.exp(sh * (th - x))))

    def _cat_ll(y, u, dt, q, r, prm, cat):
        Q, R = q**2 * dt, r**2

        def step(carry, inp):
            x, P = carry
            yt, ut = inp
            s = jnp.sqrt(P)
            f1 = _f(x + s, ut, dt, *prm, cat)
            f2 = _f(x - s, ut, dt, *prm, cat)
            xp = 0.5 * (f1 + f2)
            Pp = 0.5 * ((f1 - xp) ** 2 + (f2 - xp) ** 2) + Q
            S = Pp + R
            e = yt - xp
            ll = -0.5 * (LOG2PI + jnp.log(S) + e * e / S)
            K = Pp / S
            return (xp + K * e, Pp - K * K * S), ll

        x0 = y[:, 0]
        P0 = jnp.ones_like(x0)
        _, lls = jax.lax.scan(step, (x0, P0), (y[:, 1:].T, u[:, :-1].T))
        return lls.sum()

    _cat_ll_jit = jax.jit(_cat_ll, static_argnums=(6,))
    _cat_ll_grad = jax.jit(jax.value_and_grad(_cat_ll, argnums=(3, 4, 5)), static_argnums=(6,))
except ImportError:  # pragma: no cover
    jax = None
