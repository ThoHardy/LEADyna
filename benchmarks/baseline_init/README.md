# Baseline cut into chunks vs whole rest series — recovery of `tau` (2026-10-21)

## Setup

Linear model, SOUNDMODEL-sized design: rest + 5 SNR levels × 150 trials, epochs of 250
samples (−500 to +2000 ms at 100 Hz), input window 75–100, 100-step burn-in (stationary
start). 200 simulated participants, parameters drawn from the ranges of the SOUNDMODEL
fits: `tau` log-uniform in (2, 13), process noise U(0.2, 0.8), measure noise
U(0.15, 0.8), weights up to U(0.05, 0.45). Baseline (category 0) built three ways:

- `full`: whole catch epochs (150 × 250), no pre-stimulus;
- `chunks`: catch epochs cut into 50-sample chunks (750 × 50);
- `chunks+pre`: the chunks + the 50-sample pre-stimulus of every stimulus trial
  (1500 × 50), the SOUNDMODEL recipe.

`raw` = latents as simulated; `bc` = baseline-corrected on the pre-stimulus window
(each trial minus its mean over −500–0 ms), as in the Sergent et al. 2021 preprocessing.

## Results (`tau` bias / mean absolute relative error, n = 200)

| | full | chunks | chunks+pre |
|---|---|---|---|
| raw, LEAD init (`observation`) | −2.1 % / 3.1 % | −8.9 % / 9.1 % | −9.1 % / 9.2 % |
| raw, `stationary` init | +0.1 % / 2.5 % | +0.2 % / 2.5 % | −0.0 % / 1.7 % |
| bc, LEAD init | +16.1 % / 16.2 % | +7.2 % / 10.7 % | −14.9 % / 14.9 % |
| bc, `stationary` init | +19.0 % / 19.0 % | +18.9 % / 18.9 % | −6.1 % / 6.2 % |

With the LEAD init, short chunks bias `tau` downwards and the pre-stimulus data do not
help (bias-dominated). With the stationary init (now the default) the bias vanishes and
chunks + pre-stimulus is the best estimator (better than `full` in 63 % of participants,
Wilcoxon p = 5e-9). Baseline correction biases every construction.

`recovery_chunks.py` (clever_fit_linear; written before the change, so wrap `run` in `leadyna.use_baseline_init("observation")` to reproduce the LEAD-init rows), `stationary_init.py`
(exact Kalman MLE of the rest parameters under both inits), `analyze.py`,
`analyze_st.py`.
