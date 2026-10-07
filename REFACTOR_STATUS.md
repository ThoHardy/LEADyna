# LEADyna refactor — status & resume guide

*As of 2026-10-07 (Phase 3b complete: EEG frontend generalized, clever fits on windowed data, fast likelihood engine). Keep this file in the repo (e.g. `docs/REFACTOR_STATUS.md`) and update the checkboxes as you go. It, plus `ROADMAP.md`, is what a future session reads to pick up where we left off.*

## What this is

Turning the prototype `ThoHardy/LEAD` into **LEADyna** — a pip-installable, modality-agnostic Python toolbox: a core (models + UKF likelihood + fitting + dynamics analysis) that consumes pre-extracted latent time-series, plus swappable per-modality *frontends* that produce that format. EEG (MNE) is the only frontend for now; iEEG / infant-EEG / fMRI come later by writing new frontends against the same contract.

## Decisions locked in

- **Layered architecture**: modality-agnostic core + swappable frontends. Only the EEG frontend for now.
- **Core entry point = pre-extracted latent series** (`dict[category -> (n_trials, n_timesteps)]`). Decoding is a frontend job.
- **Foundation first**: clean installable package, honest docs, bug fixes, tests, one example — before multi-modality + analysis-suite migration.
- **Colleagues now, PyPI-ready later.**
- **License: MIT.** `.gitignore`: GitHub "Python" template.
- **`dt` is explicit but defaults to index units (`dt=1.0`).** The core no longer *silently* assumes `dt=1`; `LatentSeries` carries it. Default `1.0` reproduces historical numerics, so existing fits and the leadyna-vs-lead equivalence are preserved; physical-seconds is an opt-in that rescales `tau`/noise.
- **Import name stays `leadyna`, kept parallel to the old `lead`** so the same notebooks can be run under both and diffed for equivalence.

## Done

- [x] **Phase 0 — hygiene**: MIT `LICENSE` + Python `.gitignore` (added via GitHub UI).
- [x] **Phase 1 — package skeleton** (delivered as `LEADyna_skeleton.zip`, unpacked into the repo):
  - `src/`-layout package `src/leadyna/` with the four modules (`model.py`, `fitting_tools.py`, `dataprocess.py`, `visual.py`) copied **verbatim** — a pure rename, no logic change (so any `leadyna`-vs-`lead` notebook difference is attributable to packaging alone).
  - `pyproject.toml`: hatchling backend, `license = "MIT"`, dependency **extras** split — core (numpy/scipy/filterpy/joblib) + `[eeg]` (mne, scikit-learn) + `[viz]` (matplotlib, seaborn) + `[dev]` (pytest, ruff, nbstripout).
  - `__init__.py`: frontends load **lazily**, so the core imports with only its four core deps (an fMRI colleague never needs MNE). The one deliberate packaging change vs `lead`; changes no numbers.
  - `README.md`, `py.typed`, `tests/`, `examples/` placeholders.
  - **Verified**: builds; core imports without mne/pandas/seaborn; UKF log-likelihood == exact Kalman log-likelihood to 1.7e-13.
- [x] **Repo hygiene**: added `.gitattributes` (`eol=lf`) and normalized CRLF churn in `model.py` / `fitting_tools.py` / `.gitignore` / `LICENSE`, so line endings stop polluting diffs.
- [x] **First real tests** in `tests/` (run with `pytest`):
  - `test_likelihood.py`: UKF log-likelihood == exact analytic Kalman (linear model), and `LatentSeries` routing reproduces the dict-pair result bit-for-bit.
  - `test_contract.py`: `LatentSeries` validation (bad shapes, missing/2nd-dim<2, mismatched categories, non-finite, bad `dt` type/value, stray labels) and the model's route-through errors.
- [x] **Phase 2 — `LatentSeries` contract**: new `src/leadyna/datasets.py` with a validated `LatentSeries` dataclass (states / inputs / `dt` / `category_labels` / `metadata`). `fit`, `loglikelihood` and `loglikelihood_kalman` now accept **either** a `LatentSeries` (its `dt` is adopted) **or** the legacy dict pair — fully backward-compatible. The 7 byte-identical per-model `loglikelihood` methods were consolidated onto the base class (`_make_fx` is now the abstract hook); `fit` names `method="L-BFGS-B"` explicitly (matching the README) and no longer uses a mutable default arg. Exported `LatentSeries` from the top-level package.
- [x] **Fitting ergonomics**: `fit()` (and `loglikelihood`) now accept `n_jobs` / `batch_size`, forwarded to the UKF engine (default unchanged at `n_jobs=8`). Pass `n_jobs=1` for small/local fits — measured ~13x faster on 6 trials (0.7s vs 9.5s) with identical recovery, and no loky worker spawn. A `test_fit.py` parameter-recovery + route-equivalence test covers this.

- [x] **Phase 3 — de-EEG the core**: the modality-agnostic core no longer hard-codes 7 categories.
  - `n_categories` is now a constructor argument on every stratified model; weights `w0..w{n-1}` (and `g0..g{n-1}` for gain modulation) are built programmatically via `BaseLEADModel._register_stratified`, which also **validates** supplied weights and rejects stray/typo keys. Default `n_categories=7` reproduces the original SNR design **bit-for-bit** (guarded by a test).
  - `baseline_category` (default 0) is now an explicit, overridable attribute instead of a silent "category 0 is resting" assumption. `n_categories` / `baseline_category` are keyword-only, so a stray positional weight fails loudly rather than silently landing in `w0`.
  - **Public class names** (clean while `0.x`): `LEAD_abstract`→`BaseLEADModel`, `StratifiedLinear`→`LinearLEAD`, `NonLinear1`→`SigmoidFeedbackLEAD`, `StratifiedNonLinear1`→`StratifiedSigmoidFeedbackLEAD`, `NonLinear2`→`AffineFeedbackLEAD`, `StratifiedNonLinear2`→`StratifiedAffineFeedbackLEAD`, `GainModulation`→`GainModulationLEAD`, `StratifiedGainModulation`→`StratifiedGainModulationLEAD`. The old names remain as **deprecated aliases** (a PEP 562 `model.__getattr__` resolves them to the new class and emits a `DeprecationWarning`), so existing SOUNDMODEL notebooks keep running. The public classes are also now importable straight from the top level (`leadyna.LinearLEAD`).
  - **EEG assumptions moved out of the core**: `dataprocess.py` became the `frontends/` subpackage (`leadyna.frontends.eeg_mne`); the hard-coded 64 channels are now inferred from the data and `decimate(5)` is a `decimate_factor` argument. `leadyna.dataprocess` stays as a deprecated shim. The UKF engine and every model's physics are untouched.
  - **Tests: 34 pass** (was 18). New `test_categories.py` (free `n_categories`: 7≡explicit-7, UKF≡Kalman on 3 categories, LatentSeries round-trip on 4, weight recovery, baseline exposure, stray-weight rejection) and `test_aliases.py` (every deprecated alias warns and resolves). Existing tests migrated to the new names.

- [x] **Phase 3b — generalize to new EEG datasets** (2026-10-06; motivated by the developmental ba/da-in-noise dataset: adults, 11-month-olds, 4-month-olds, 13 electrodes, EEGLAB format):
  - **Frontend**: new `decode_latent(epochs, train_window, *, condition="snr", baseline_label=0, present_label=None, cv="blocknumber", target_sfreq=100.0)` in `frontends/eeg_mne.py`. Accepts any `mne.BaseEpochs` (Epochs, EpochsArray, EEGLAB/FIF) or a path; never modifies its input; windows in seconds; integer-like labels (string labels are cast); **category 0 = baseline, hard-coded**, other labels sorted → 1..K; `cv` = block column (leave-one-block-out) or int (stratified k-fold); resampled to `target_sfreq` (decimate on integer ratios = LEAD numerics, FFT resample otherwise). Returns a `LatentSeries` with `inputs=None` and metadata `times`, `sfreq`, `trial_index` (original epoch index of every row; rows keep epoch order whatever the folds, so random block numbers are safe), `label_to_category`.
  - `STG` kept as a deprecated wrapper (ms windows, baseline label 1, dict output, also accepts in-memory epochs): **equal to the LEAD STG to 1e-9** on synthetic Sergent-like data (test). `substract_pattern` and `sort_by_snr=False` removed.
  - `LatentSeries.inputs` may be `None`; `with_inputs()` and `times` added; fitting a LatentSeries without inputs raises a clear error.
  - **Backend**: `clever_fit_*` take **already-windowed** data (category 0 = baseline segments, others = constant-input window); passing `input_start_index`/`input_stop_index` reproduces LEAD (DeprecationWarning). Options `bounds` (dict, defaults = SOUNDMODEL values), `threshold_grid`, `sharpness`, `n_jobs`. Non-contiguous categories supported (missing level → weight stays 0). `dt` of a LatentSeries is honoured. **Bit-identical to LEAD** for linear, null, gain-modulated and fixed-gain fits (`pytest -m slow`, ~6 min).
  - `pandas` added to the `[eeg]` extra; `LogisticRegression(penalty=...)` dropped (deprecated in scikit-learn 1.8, same L2 default).
  - README rewritten as a tutorial (toy EEG → `decode_latent` → input hypothesis → three fits → held-out comparison); runnable copy in `examples/toy_example.py`.
  - Tests: 50 fast + 3 slow (`tests/_legacy_stg.py` and `tests/_legacy_fitting_tools.py` are frozen LEAD oracles).

- [x] **Fast likelihood engine** (2026-10-07): `compute_ukf_loglikelihood_fast` in `model.py`, numpy, vectorized over trials (sequential in time). Default `BaseLEADModel.engine = "fast"`; `"filterpy"` (the LEAD engine) selectable per call (`engine=`), per model (`model.engine`) or globally (`leadyna.use_engine("filterpy")`, which also covers the clever fits). Only the base class changed. Equal to filterpy to ~1e-11 on the log-likelihood for all 7 concrete models, with constant and trial-varying inputs, and to the exact Kalman for the linear model (`tests/test_engines.py`). ×550 per evaluation on infant-sized data (4 levels × 150 trials); README toy example 8 min → 46 s with the full search. Fitted parameters differ from filterpy by ~1e-5 (finite-difference gradients); **SOUNDMODEL numbers are reproduced bit-for-bit only with `use_engine("filterpy")`**. Clever-fit LEAD-equivalence tests now run under both engines (fast: seconds; filterpy: `pytest -m slow`). Benchmark of options A (numpy, chosen), B (numba), C (JAX) in `benchmarks/ukf_engines/`.
- [x] **Trial-varying inputs at full speed** (2026-10-07): `StratifiedGainModulationLEAD` now computes `g{int(category*u)}` element-wise over trials (`_gain_for`, also used by `nonlinearity`, which previously applied trial 0's input to every simulated trial in `measure_simulations`; out-of-range `u` now raises instead of failing on a missing attribute). The fast engine passes the input vector to `_make_fx` (`BaseLEADModel.vector_inputs = True`; custom scalar-`u` models set it to False to get the grouping fallback). Numerics unchanged (fast = filterpy to 0 on shared inputs; LEAD equivalence under filterpy re-checked).

## Environment (important lesson)

Install into an **isolated environment**, never the conda `base` env. Unpinned `numpy>=1.24` let pip pull numpy 2.x into base and break other tools (gensim, numba need numpy < 2). The fix is isolation, not upper-capping deps — libraries declare lower bounds only. `leadyna` itself runs fine under numpy 2.x.

Recipe (venv, works in Git Bash without conda):

```bash
cd /c/Users/thoma/LEADyna
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e ".[eeg,viz,dev]"
python -m pip install ipykernel
python -m ipykernel install --user --name leadyna --display-name "Python (leadyna)"
```

Then select the "Python (leadyna)" kernel in notebooks. (conda-env equivalent: run the same `pip install -e` inside a `conda create -n leadyna python=3.12` env from the Anaconda Prompt.)

To repair base if numpy got bumped there: `python -m pip install "numpy==1.26.4" "scipy==1.13.1" "contourpy==1.2.0"` and `python -m pip uninstall leadyna mne`.

## Next steps (detail in `ROADMAP.md`)

- [ ] Confirm a SOUNDMODEL notebook reproduces under `import leadyna as lead` (equivalence check).
- [x] UKF-vs-Kalman check is now the first real test in `tests/`. **Still TODO:** add GitHub Actions CI (Phase 5).
- [x] **Phase 2 — `LatentSeries` contract**: done (see **Done** above).
- [x] **Phase 3 — de-EEG the core**: done (see **Done** above) — free `n_categories`, explicit `baseline_category`, public `*LEAD` names + deprecated aliases, EEG assumptions moved to `frontends/eeg_mne.py`.
- [x] `clever_fit_gainmodul(n_thresholds=5)`: number of initial thresholds, evenly spaced over the threshold bounds (2026-10-07; default reproduces LEAD; explicit `threshold_grid` overrides it).
- [ ] **Convergence of the clever fits** (2026-10-07, `benchmarks/n_loops/`): fixed gain is far from converged at `n_loops=2` (≈46 train / 8 test nats below the joint optimum on infant-sized toy data), gain modulated converges in 1 loop but never refits `tau` (biased by short baseline segments). A final joint polish (one L-BFGS-B over all free parameters, 5–9 s) beats 8 loops for both. Decide whether to add it as an option / default.
- [ ] **JAX engine (option C)** — revisit when models get bigger (multi-dimensional latents, many parameters): exact autodiff gradients for L-BFGS-B. See `benchmarks/ukf_engines/README.md`.
- [ ] **CV in SOUNDMODEL notebooks** (`5CV_*`): `KFold` runs over `arange(min n_trials across categories)`, so trials beyond the smallest category are never used. Fix when migrating CV into `compare.py` (split each category separately).
- [ ] **Phase 4 — dynamics**: expose `model.drift()` / `drift_deriv()`; migrate bifurcation-probability + metastability score from SOUNDMODEL into `lead/dynamics.py`.
- [ ] **Phase 5 — compare**: migrate CV / Bayesian model selection / OVL-STD-Wasserstein into `lead/compare.py`.
- [ ] **Phase 6 — one worked example** notebook (EEG `.fif` → fit → compare → bifurcation → plot).

## Decisions taken in Phase 3

- **Weights**: kept per-category attributes `w0..w{n-1}` (dynamically generated from `n_categories`) rather than a vector, so notebooks that pass `w1=...` or read `.w3` keep working unchanged.
- **Baseline**: kept the designated baseline category, but made it explicit and overridable (`baseline_category=0`).
- **Class names**: renamed to public `*LEAD` names, with the old names kept as deprecated aliases (non-breaking).
- **Category vs continuous input**: kept discrete `category`, only freed its cardinality (per the foundation-first call). Continuous `I(t)` deferred until a real dataset demands it.

## Still open (later phases)

- Author metadata in `pyproject.toml`: add email / ORCID / lab as second copyright holder? (cheap, do before first PyPI release.)
- Confirm a SOUNDMODEL notebook reproduces end-to-end under the new names + aliases (equivalence check on real data).

## How to resume in a new Claude Cowork conversation

Claude's memory persists across Cowork conversations, so a new chat already knows this project (it's filed under "LEAD toolbox / LEADyna"). To resume efficiently:

1. **Connect the repo folder** `C:\Users\thoma\LEADyna` to the new session (so Claude reads the actual current code), or rely on the public repo at `github.com/ThoHardy/LEADyna`.
2. **Open with a message like:**

   > Continue the LEADyna refactor. Read `ROADMAP.md` and `docs/REFACTOR_STATUS.md` in the repo (or github.com/ThoHardy/LEADyna) and pick up from the next unchecked step — I want to do Phase 2, the LatentSeries contract.

3. Keep `ROADMAP.md` and this file committed and current — they are the durable source of truth; the conversation is not.

Carry-over artifacts from the setup session: `LEAD_refactor_roadmap.md` (full phase plan) and `LEADyna_skeleton.zip` (already unpacked into the repo).
