# Goodness of fit vs `n_loops` — fixed gain vs gain modulated (2026-10-07)

## Setup

Data simulated from each model, infant-like design: rest + 4 levels, 100 trials per
level (80 train / 20 test), input window 40 steps, baseline = rest cut into 20-step
segments + 20-step pre-stimulus segments (27 680 train / 6 920 test samples). Truths:
tau 10, noises 0.3, threshold 1, sharpness 5, w = .02–.08; gain-modulated g = .05–.20,
fixed gain 0.15. Seeds 0–3. `loops_exp.py` runs instrumented copies of
`clever_fit_nonlinear1` / `clever_fit_gainmodul` that snapshot every candidate after
each loop (checked equal to the library for `n_loops` = 1, 2). "Joint polish" = one
L-BFGS-B over every free parameter (sharpness fixed) from the loop-8 solution.

Indicator: log-likelihood per sample minus that of the generating model, in
milli-nats/sample (×1e-3), on train and held-out test data. Positive values are possible
because the UKF likelihood is an approximation, so the true parameters do not maximize it.

## Results (mean over 4 seeds)

| loop | FG fit, FG data: train | test | GM fit, GM data: train | test |
|---|---|---|---|---|
| 1 | −1.97 | −1.66 | +0.57 | +0.27 |
| 2 (default) | −1.20 | −1.04 | +0.58 | +0.24 |
| 3 | −0.74 | −0.69 | +0.58 | +0.23 |
| 4 | −0.44 | −0.46 | +0.58 | +0.22 |
| 6 | −0.07 | −0.18 | +0.58 | +0.22 |
| 8 | +0.13 | −0.04 | +0.58 | +0.22 |
| joint polish | +0.46 | +0.08 | +0.66 | +0.27 |

Cost of 8 loops: ~70–85 s (FG), ~40–55 s (GM); joint polish: 5–9 s.

## Findings

- **Fixed gain is not converged at the default `n_loops=2`**: 1.66e-3 nats/sample below the
  joint optimum on train (≈ 46 nats in total) and 1.1e-3 on test (≈ 8 held-out nats). It is
  still climbing at loop 8. Cause: `tau` starts from the resting-state OU fit, which is
  biased when the fixed-gain feedback is active at rest (14.5 instead of 10 here), and the
  alternating steps crawl along a `tau`–`gain` ridge (tau 14.5 → 13.0 → … → 10.1 by loop 8).
- **Gain modulated is converged after one loop**; extra loops cost time for nothing. But it
  never refits `tau`, so it keeps the resting-state estimate, which is biased by the short
  baseline segments (≈ 8.0–8.3 for a true 10 with 20-step segments; 9.2 with 40; 9.5
  with 120) and, on fixed-gain data, by the feedback at rest (17.4).
- **A single joint polish beats 8 loops** for both models, on train and test, at the cost
  of about one loop.
- `tau` and the gain are only weakly identified jointly: the polish improves the
  likelihood while moving `tau` away from the truth on fixed-gain data (8.6) — parameter
  recovery of `tau` alone is not a good goodness-of-fit indicator.
- Consequence for model comparison: with the defaults, the fixed-gain model is under-fitted
  relative to the gain-modulated one, which biases cross-validated comparisons towards the
  gain-modulated model.

Re-run: `python loops_exp.py run 100 0,1 8 rows.json` then `python analyze.py rows.json`.
