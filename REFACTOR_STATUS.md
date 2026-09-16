# LEADyna refactor — status & resume guide

*As of 2026-09-16. Keep this file in the repo (e.g. `docs/REFACTOR_STATUS.md`) and update the checkboxes as you go. It, plus `ROADMAP.md`, is what a future session reads to pick up where we left off.*

## What this is

Turning the prototype `ThoHardy/LEAD` into **LEADyna** — a pip-installable, modality-agnostic Python toolbox: a core (models + UKF likelihood + fitting + dynamics analysis) that consumes pre-extracted latent time-series, plus swappable per-modality *frontends* that produce that format. EEG (MNE) is the only frontend for now; iEEG / infant-EEG / fMRI come later by writing new frontends against the same contract.

## Decisions locked in

- **Layered architecture**: modality-agnostic core + swappable frontends. Only the EEG frontend for now.
- **Core entry point = pre-extracted latent series** (`dict[category -> (n_trials, n_timesteps)]`). Decoding is a frontend job.
- **Foundation first**: clean installable package, honest docs, bug fixes, tests, one example — before multi-modality + analysis-suite migration.
- **Colleagues now, PyPI-ready later.**
- **License: MIT.** `.gitignore`: GitHub "Python" template.
- **Import name stays `leadyna`, kept parallel to the old `lead`** so the same notebooks can be run under both and diffed for equivalence.

## Done

- [x] **Phase 0 — hygiene**: MIT `LICENSE` + Python `.gitignore` (added via GitHub UI).
- [x] **Phase 1 — package skeleton** (delivered as `LEADyna_skeleton.zip`, unpacked into the repo):
  - `src/`-layout package `src/leadyna/` with the four modules (`model.py`, `fitting_tools.py`, `dataprocess.py`, `visual.py`) copied **verbatim** — a pure rename, no logic change (so any `leadyna`-vs-`lead` notebook difference is attributable to packaging alone).
  - `pyproject.toml`: hatchling backend, `license = "MIT"`, dependency **extras** split — core (numpy/scipy/filterpy/joblib) + `[eeg]` (mne, scikit-learn) + `[viz]` (matplotlib, seaborn) + `[dev]` (pytest, ruff, nbstripout).
  - `__init__.py`: frontends load **lazily**, so the core imports with only its four core deps (an fMRI colleague never needs MNE). The one deliberate packaging change vs `lead`; changes no numbers.
  - `README.md`, `py.typed`, `tests/`, `examples/` placeholders.
  - **Verified**: builds; core imports without mne/pandas/seaborn; UKF log-likelihood == exact Kalman log-likelihood to 1.7e-13.

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
- [ ] Turn the UKF-vs-Kalman check into the first real test in `tests/`; add GitHub Actions CI.
- [ ] **Phase 2 — `LatentSeries` contract**: a small validated container (states / inputs / `dt` / category labels) as the core↔frontend boundary; make `dt` explicit end-to-end.
- [ ] **Phase 3 — de-EEG the core**: make `n_categories` a parameter (drop the hard-coded `w0..w6` / `range(7)`); move 64-channel / decimate / metadata assumptions into `frontends/eeg_mne.py`; rename classes for clarity while still `0.x`.
- [ ] **Phase 4 — dynamics**: expose `model.drift()` / `drift_deriv()`; migrate bifurcation-probability + metastability score from SOUNDMODEL into `lead/dynamics.py`.
- [ ] **Phase 5 — compare**: migrate CV / Bayesian model selection / OVL-STD-Wasserstein into `lead/compare.py`.
- [ ] **Phase 6 — one worked example** notebook (EEG `.fif` → fit → compare → bifurcation → plot).

## Open questions still to decide

- Core abstraction: keep discrete `category`, or take a continuous input `I(t)` and let frontends bin?
- Does the core need a designated "baseline/resting" category, or should that be a frontend flag?
- Public class names (`LEAD_abstract` → `BaseLEADModel`, `Stratified*` → clearer names?).
- Author metadata in `pyproject.toml`: add email / ORCID / lab as second copyright holder?

## How to resume in a new Claude Cowork conversation

Claude's memory persists across Cowork conversations, so a new chat already knows this project (it's filed under "LEAD toolbox / LEADyna"). To resume efficiently:

1. **Connect the repo folder** `C:\Users\thoma\LEADyna` to the new session (so Claude reads the actual current code), or rely on the public repo at `github.com/ThoHardy/LEADyna`.
2. **Open with a message like:**

   > Continue the LEADyna refactor. Read `ROADMAP.md` and `docs/REFACTOR_STATUS.md` in the repo (or github.com/ThoHardy/LEADyna) and pick up from the next unchecked step — I want to do Phase 2, the LatentSeries contract.

3. Keep `ROADMAP.md` and this file committed and current — they are the durable source of truth; the conversation is not.

Carry-over artifacts from the setup session: `LEAD_refactor_roadmap.md` (full phase plan) and `LEADyna_skeleton.zip` (already unpacked into the repo).
