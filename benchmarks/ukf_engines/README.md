# UKF likelihood engines — benchmark (2026-10-07)

Kept for later: revisit **option C (JAX)** when LEADyna moves to bigger models
(multi-dimensional latents, many more parameters), where exact gradients matter most.

## Question

Can the UKF log-likelihood be vectorized, given that the filter is sequential in time
(each step depends on the previous one), inputs vary over time, and categories differ?

Yes: the recursion stays sequential in time, but trials are independent, so all trials
of a category advance together. Time-varying inputs are a vector `u[:, t]` per step;
categories are an outer loop (or per-trial parameter arrays). In 1-D with Julier sigma
points and kappa = 0, the sigma set is {x, x ± sqrt(P)} with weights {0, ½, ½}: one UKF
step is two evaluations of the transition function.

## Setup

Toy data simulated from `StratifiedGainModulationLEAD` (tau 10, noises 0.3, threshold 1,
sharpness 5, w = .02–.08, g = .05–.2), infant-like design: rest + 4 SNR levels, epochs
-0.2–1.0 s at 100 Hz, input window 0.4–0.8 s (40 steps), baseline = rest cut into
20-step segments + 20-step pre-stimulus of every stimulus trial. With 150 trials per
level: 2 100 sequences, 54 000 samples. Cloud container, 2 cores, single process.

## Cost of one log-likelihood evaluation

| Engine | 5 trials/level | 30 trials/level | 150 trials/level | |LL − filterpy| |
|---|---|---|---|---|
| filterpy (LEAD engine) | 0.9 s | 5.8 s | 26 s | — |
| A. numpy, vectorized over trials | 17 ms | 14 ms | 47 ms (×550) | 7e-12 |
| B. numba, compiled scalar loops | 0.24 ms | 0.8 ms | 8 ms (×3 100) | 4e-10 |
| C. JAX, `lax.scan` over time + vectorized trials | 2.4 ms | 3.0 ms | 6 ms (×4 500) | 7e-12 |

JAX value + full gradient (autodiff) costs ~1.15× one value (0.16 vs 0.14 ms on one
category), versus k + 1 values for the finite-difference gradients L-BFGS-B uses now.
Autodiff dLL/dtau matched finite differences (−0.49995 vs −0.49996).

## End-to-end fits

- Realistic data (150 trials/level), engine A, default search (5 thresholds × 2 inits,
  `n_loops=2`): linear + fixed gain (~27 s) + modulated gain (~20 s) ≈ 1 min per subject.
  At the measured per-evaluation ratio, the filterpy engine would need hours per subject
  and per fold.
- Engine A vs filterpy, same clever fits on 5 trials/level (light search): 8 s vs 6 min;
  fitted parameters differ by ≤ 4e-5 (finite-difference gradients amplify the 1e-12
  likelihood differences), so not bit-identical.
- README toy example (40 trials/level, default search): 8 min (filterpy, 20 trials/level
  and light search) → 46 s (engine A, full search).

## Decision

- **A is integrated** as `engine="fast"` (default) in `BaseLEADModel`
  (`leadyna.model.compute_ukf_loglikelihood_fast`); `"filterpy"` stays available for
  bit-for-bit LEAD / SOUNDMODEL reproduction (`leadyna.use_engine("filterpy")`). First
  version (base class only): the engine passed a scalar `u` and grouped trials by input
  value, so `StratifiedGainModulationLEAD`'s scalar `int(category * u)` worked unchanged —
  but with a different input on every trial the speed-up fell from ×580 to ×25
  (150 groups per step). Second version: `StratifiedGainModulationLEAD` computes its gain
  index element-wise (`_gain_for`), the engine passes the input vector directly
  (`vector_inputs = True`), and trial-varying inputs cost the same as shared ones
  (26 ms vs 26 ms; grouping fallback: 980 ms). The grouping path stays for custom
  models that set `vector_inputs = False`.
- **B (numba)** rejected: fastest on small data but every model's physics must be
  duplicated in a compiled kernel.
- **C (JAX)** for later: exact gradients and the best scaling, but models must be
  rewritten in `jax.numpy` and the dependency is heavy. Worth it for larger models.

## Files

- `engines.py`: prototypes of A, B, C (B and C need `numba` / `jax[cpu]`, not LEADyna
  dependencies).
- `bench_eval.py`: the per-evaluation benchmark above
  (`python bench_eval.py` from this folder).
