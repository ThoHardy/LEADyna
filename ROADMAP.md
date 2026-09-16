# Refactoring LEAD into a reusable toolbox — roadmap & design notes

*A working reference for turning `ThoHardy/LEAD` into a pip-installable, modality-agnostic Python library, with `SOUNDMODEL` as its first "application". Written to be dropped into the repo (e.g. as `ROADMAP.md`) and revised as you go.*

Decisions taken up front (they shape everything below):

- **Layered architecture**: a modality-agnostic **core** (models + UKF + fitting + dynamics analysis) plus swappable **frontends** that turn raw data into the core's input format. Only the **EEG frontend** is built for now.
- **Entry point of the core = pre-extracted latent time-series.** Decoding is a frontend concern, not a core concern.
- **Foundation first**: clean installable package, honest docs, fixed bugs, tests, one worked example. Multi-modality frontends and the full analysis-suite migration come in a second pass.
- **Colleagues now, PyPI-ready later.**

---

## 1. The single most important idea

Your core is *already* modality-agnostic and you may not have noticed. Look at what `compute_ukf_loglikelihood`, `LEAD_abstract`, and `fit` actually consume:

```
state_series : dict[category -> ndarray(n_trials, n_timesteps)]
input_series : dict[category -> ndarray(n_trials, n_timesteps)]
```

That is just numbers. Nothing in `model.py` knows about EEG, electrodes, MNE, or the brain. **All** the EEG-specificity lives in `dataprocess.STG` (which reads `.fif`, decimates, trains logistic regressions) and in a few hard-coded assumptions that leaked into `fitting_tools.py` (the "7 categories", the `75:100` window). 

So the refactor is not "make the maths general" — the maths is general. The refactor is:

1. **Name and freeze the contract** between core and frontends (the `state_series`/`input_series` format), so a colleague with iEEG or fMRI knows exactly what to hand you.
2. **Pull the EEG assumptions out of the core** and into the EEG frontend, where they belong.
3. **Package it** so `pip install lead` works and `import lead` gives a clean, documented API.

Everything else is refinement. Keep this framing when you feel lost in the details.

---

## 2. Target architecture

```
lead/                          # the installable package (modality-agnostic unless noted)
├── __init__.py                # curated public API only
├── core/
│   ├── ukf.py                 # compute_ukf_loglikelihood (the engine)
│   ├── base.py                # BaseLEADModel (abstract) + shared fit/simulate/params
│   └── models.py              # Linear, GainModulation, … concrete models
├── fitting.py                 # "clever" staged-fitting strategies (model-generic)
├── dynamics.py                # NEW: f(x) analysis — equilibria, bifurcation, metastability
├── compare.py                 # NEW: CV, BIC/AIC, Bayesian model selection, distances
├── viz.py                     # trajectories, densities, colormap (optional dep)
├── datasets.py                # the LatentSeries container + validation of the contract
└── frontends/
    └── eeg_mne.py             # the current STG, generalized (optional deps: mne, sklearn)
```

The `frontends/` folder is where "layered but EEG-only for now" lives: today it holds one file; tomorrow `ieeg.py` and `fmri.py` sit beside it, each with the single job of producing the same `LatentSeries` object. The core never imports from `frontends/`; frontends import from the core. That one-directional dependency is what makes it a real library instead of a pile of scripts.

### 2.1 The contract (this is the deliverable that unlocks the other modalities)

Define one small data structure — call it `LatentSeries` — that *is* the boundary. Even a lightweight class or a validated dict is fine. It should carry:

- `states`: `dict[category -> (n_trials, n_timesteps)]` — the latent variable `s(t)` per trial.
- `inputs`: `dict[category -> (n_trials, n_timesteps)]` — the exogenous drive `I(t)` (currently your binary stim-on/off).
- `dt`: the timestep in physical units (seconds). **Make this explicit and required** — it is currently a silent `self.dt = 1` default in the model, which is a bug waiting to happen the moment someone uses non-10ms data (see §4).
- `category_labels` / metadata: what each category means (SNR level, condition…), so plots and reports aren't hard-coded to "0=rest, 1–6=SNR".

Once this object exists and is validated in one place, "supporting iEEG" means "write a frontend that returns a `LatentSeries`", and the whole downstream stack (fit, compare, bifurcation) works unchanged. That is the pay-off of the layering.

### 2.2 A caveat you should design against, not around (fMRI)

The UKF likelihood is a product over time steps; with only 2–3 timepoints per trial (fMRI, even "temporally precise" fMRI at TR≈0.5–1s) the likelihood is thin and the fit is weakly constrained. The contract still *accepts* fMRI — nothing breaks — but the honest position is: **the core runs on any ≥2-timepoint series; statistical power is a property of the data, not the library.** Put that sentence in the docs so an fMRI colleague calibrates expectations. It also argues for the model-comparison tools (§6) being modality-aware about how much they can conclude.

---

## 3. What to fix in the existing code (the "make it trustworthy" pass)

These are concrete issues I found reading the current source. None are hard; together they're the difference between "prototype" and "library a colleague trusts".

**Correctness / footguns**

- `measure_simulations` (base class): `if initial_states != None` and the model's `input_value[0]` handling in `StratifiedGainModulation.nonlinearity` use `!= None` / bare `except:` — switch to `is not None` and remove blanket excepts. The `idx = int(signal_category * inp_scalar)` gain-indexing in `StratifiedGainModulation` is a fragile hack (it multiplies category by the input scalar to pick `g{idx}`); document what it's meant to express or replace it with an explicit mapping — it will silently mis-index on any new dataset.
- `self.dt = 1` default in `BaseLEADModel.__init__`. Every physical quantity (tau in "ms", the `Q = process_noise**2 * dt` discretisation) depends on dt. Make dt come from the data (`LatentSeries.dt`), not a default, and use consistent physical units throughout.
- `StratifiedNonLinear1._param_names` carries a code comment admitting the parameter list may break `load_params` compatibility ("I forgot sharpness … IMPORTANT TO DO LATER"). Parameter (de)serialisation that can silently mis-map is exactly what a test must lock down (§7).
- `fit()` calls `scipy.optimize.minimize(...)` without naming a method; with bounds this defaults to L-BFGS-B, but the ReadMe *claims* L-BFGS-B explicitly — make it explicit in code so the claim can't drift.

**Hard-coded EEG assumptions that must move to the frontend or become parameters**

- The "7 categories" (`w0..w6`, `range(7)`) is baked into every model's `_param_names`. This is your SNR design, not a law of nature. Make the number of categories a constructor argument (`n_categories`), and build the `w`/`g` parameter vectors programmatically. This is the biggest single change for genuine reusability — infant-EEG or iEEG experiments won't have 7 SNR levels.
- `dataprocess.STG` hard-codes `.reshape(-1, 64)` (64 channels!), `.decimate(5)`, `blocknumber`/`snr` metadata columns, and leave-one-block-out CV. All legitimate for your EEG — all fatal for a 9-electrode infant cap. These belong in `frontends/eeg_mne.py` with channel count inferred from the data and the decimation/metadata names as arguments.
- `fitting_tools` hard-codes `input_start_index=75, input_stop_index=100` (your 250–500 ms window at 10 ms). Keep them as arguments (they already are) but express them in physical time via `dt`, and stop assuming category 0 is always resting-state — make "which category is the baseline" explicit.

**Performance**

- `STG` re-reads the epochs file from disk inside the block loop (`mne.read_epochs(...)` twice per block). Read once. Minor now, painful on big datasets.
- `n_jobs=8` and `batch_size=20` are hard-coded defaults; expose them and default `n_jobs=-1` (all cores) or read from an env var.

**Repo hygiene (do this early, it's cheap and high-impact)**

- The repo commits `__pycache__/`, `.ipynb_checkpoints/`, and — in SOUNDMODEL — notebooks up to **63 MB** (`Plot_late.ipynb`) with outputs embedded. Add a `.gitignore`, strip notebook outputs (`nbstripout` as a git filter, or `jupytext` to pair `.ipynb` with a plain `.py`), and consider `git filter-repo` to purge the big blobs from history before others clone. A 60 MB notebook in history makes the repo slow to clone forever.
- `ReadMe.txt` (36 KB) has already drifted from the code — it documents an `euler_maruyama()` method and `n_loops`/`feedback` arguments that don't exist in the current source. A big prose doc that lies is worse than a short one that's true. Replace it with a short `README.md` (install + 20-line quickstart) and let **docstrings** be the reference (§7). Docstrings can't drift as easily because they sit next to the code.

---

## 4. What from SOUNDMODEL belongs *in* LEAD

The test for "does it move into the library?": **does it operate purely on a model / its fitted parameters / a latent series, with nothing specific to Sergent 2021?** If yes → library. If it's paper glue (condition orchestration, figure styling, the specific 4 decoders) → stays in SOUNDMODEL.

Moves into LEAD:

- **`dynamics.py` (new, high value).** Everything about the fitted drift function `f(x)`: locate equilibria, classify them (stable/unstable/metastable via `f(x_eq)` and `df/dx`), the effective-bifurcation rule (zero-crossing change between input levels with your "≥10% of high-SNR states above the inflexion point" constraint), the **bifurcation-probability grid search** over `(tau, inflexion_point)`, and the **metastability score** `MetaScore = −min_x(f² + f'²)`. These are genuinely reusable and modality-blind — they're the scientific heart of the toolbox, currently trapped in notebooks.
  - **Prerequisite API addition**: expose the drift itself. Right now `core()` returns `x + dxdt·dt` (the next state), not `dxdt`. Add a public `drift(x, u, category) -> dxdt` and, where you can, an analytic `drift_deriv(x, u, category)` on each model. The bifurcation/metastability code needs `f` and `f'` directly; giving them a clean home also removes duplicated `f(x)` reconstruction from every notebook.
- **`compare.py` (new).** The k-fold CV loop, per-model held-out log-likelihood, BIC/AIC, group-level Bayesian model selection (Rigoux/Stephan — PEP, exceedance probabilities), and the distribution-distance metrics (OVL, STD, Wasserstein) you compute across notebooks. Today these are copy-pasted across `5CV_*`, `ModelComparison_*`, `Metastability*`, `BifurcationProbability_*` — dozens of near-identical notebooks. One tested module replaces all of them.
- **Parameter recovery** (`ParameterRecovery.ipynb`) → becomes both a `lead.validate` helper *and* part of the test suite (§7): simulate from known params, fit, assert recovery within tolerance. This is the best possible regression test for a fitting library.

Stays in SOUNDMODEL (for now):

- Behavioral prediction (the separatrix/basin readout, mixture classifiers) — still research-in-progress and coupled to the report/no-report design. Promote it later once it stabilises.
- The Sergent-2021 orchestration: which 4 decoders, the Active/Passive × Early/Late loop, paper figures. SOUNDMODEL becomes a *thin* consumer of LEAD: extract → fit → compare → plot, with the heavy logic imported, not redefined.

A good milestone: **SOUNDMODEL should shrink dramatically.** If it doesn't, logic that should have moved didn't.

---

## 5. Packaging & open-source mechanics (the "learn how to build a toolbox" part)

Here's the modern (2026) minimal-but-real setup, roughly in the order a colleague's `pip install` exercises it.

**Project layout — use the `src/` layout.**

```
LEAD/
├── pyproject.toml          # the one config file that matters
├── README.md               # short: what it is, install, 20-line quickstart
├── LICENSE                 # pick one (see below)
├── src/lead/…              # the package (importing works only after install → catches packaging bugs)
├── tests/                  # pytest
├── examples/               # cleaned notebooks / scripts (NOT shipped in the wheel)
└── .github/workflows/ci.yml
```

The `src/` layout matters more than it looks: it forces you to actually install the package to import it, so "works on my machine because I'm cd'd into the folder" bugs surface immediately — which is exactly the failure mode when a colleague installs it.

**`pyproject.toml` — the single source of truth.** One file declares the build backend, metadata, dependencies, and tool config. A build backend like `hatchling` or `setuptools` is fine; `hatchling` is the low-ceremony default now. It holds:

- **name, version, description, authors, license, python-requires** (pin `>=3.11` — you use `dict[int, np.ndarray]` builtins syntax already).
- **Dependencies, split into core vs optional "extras"** — this is the key trick for a multi-modality library. Core stays tiny; heavy/modality-specific deps go in extras a colleague opts into:
  - core: `numpy`, `scipy`, `filterpy`, `joblib`
  - `[eeg]`: `mne`, `scikit-learn` (only the EEG frontend needs these)
  - `[viz]`: `matplotlib`, `seaborn`
  - `[dev]`: `pytest`, `ruff`, `nbstripout`
  - So `pip install lead` is lean; `pip install "lead[eeg,viz]"` gets the EEG pipeline. An fMRI colleague never installs MNE.
- **A note on `filterpy`**: it's lightly maintained and is your only "exotic" dependency. Keep it for now (rewriting the UKF is a distraction), but your `loglikelihood_kalman` exact scalar filter shows you could vendor a ~40-line UKF later and drop the dependency entirely. Flag it as tech-debt, don't act on it yet.

**Versioning.** SemVer, start at `0.1.0`. While `0.x`, you're allowed to break the API between minor versions — say so in the README, and use this freedom now to fix the naming (`LEAD_abstract` → `BaseLEADModel`, etc.) before anyone depends on it. Expose `lead.__version__`.

**License.** For a science library that you *want* colleagues and others to adopt, a permissive license is the norm — **MIT** (shortest) or **BSD-3-Clause** (what MNE, scikit-learn, NumPy use, so it composes cleanly with your stack). Pick one and add the `LICENSE` file on day one; retrofitting a license after others contribute is a headache. Avoid GPL unless you specifically want copyleft — it complicates others building on top.

**The public API (`__init__.py`).** Decide deliberately what `import lead` exposes. Your current `__init__` re-exports `colormap` and `STG` — fine as a start, but curate: export the model classes, `fit`/the fitting strategies, the `LatentSeries` container, and the analysis entry points; keep internals private (leading underscore or just not re-exported). The public API is a *promise*; the smaller it is, the less you're locked into.

**Docstrings + docs.** Adopt one docstring style (**NumPyDoc** — matches your ecosystem) and write them as you refactor. That alone is "documentation" for v0.1. Later, MkDocs (with `mkdocstrings`) or Sphinx can auto-render those docstrings into a website with almost no extra writing — but don't build the doc site until the API has settled.

**Environment / reproducibility.** `pyproject.toml` is enough for `pip`. If your lab lives in conda, also ship an `environment.yml`. Tell people the exact Python you support and test on.

**Code quality tooling (optional but cheap).** `ruff` does linting *and* formatting in one fast tool — one config block in `pyproject.toml`. Add `pre-commit` so `ruff` + `nbstripout` run automatically before each commit; this is what keeps 60 MB notebooks and style drift out of the repo without you policing it.

---

## 6. Tests & CI (what "trustworthy" concretely means)

You don't need 100% coverage; you need the **oracles** that catch the bugs that would silently corrupt a colleague's results:

- **Exact-likelihood test**: for the linear model you already have *two* likelihood paths — the UKF (`loglikelihood`) and the analytic Kalman (`loglikelihood_kalman`). Assert they agree to numerical tolerance. This one test validates the entire UKF engine for the linear case for free. (You essentially already wrote the test oracle; it just needs to live in `tests/`.)
- **Parameter recovery**: simulate from known parameters → fit → assert recovery within tolerance, per model. Catches optimisation/param-mapping regressions.
- **Serialisation round-trip**: `save_params` → `load_params` → identical params, for every model. This is the exact class of bug your `StratifiedNonLinear1` comment worries about.
- **Contract validation**: bad `LatentSeries` (mismatched shapes, missing dt, single timepoint) raises clear errors, not obscure ones deep in the UKF.

**CI**: a `.github/workflows/ci.yml` that runs `pytest` (and `ruff check`) on every push and pull request, on the Python versions you claim to support. Ten lines of YAML; it's what lets you and colleagues change code without fear.

---

## 7. Suggested order of work (phased, foundation-first)

Each phase leaves the library in a working, installable state — never a big-bang rewrite.

**Phase 0 — hygiene (half a day).** Add `.gitignore`, strip/untrack `__pycache__`, `.ipynb_checkpoints`, notebook outputs; purge the giant blobs from history; add `LICENSE`. Nothing functional, big cleanliness win.

**Phase 1 — package skeleton.** `src/lead/` layout, `pyproject.toml` with core deps + `[eeg]`/`[viz]`/`[dev]` extras, move existing modules in unchanged, get `pip install -e ".[eeg,viz]"` + `import lead` working. Replace `ReadMe.txt` with a short honest `README.md`. **Deliverable: a colleague can install it.**

**Phase 2 — define the contract.** Introduce `LatentSeries` (states/inputs/dt/labels) with validation. Route `fit`/`loglikelihood` through it. Make `dt` explicit end-to-end. **Deliverable: the core no longer assumes 10 ms or your file format.**

**Phase 3 — de-EEG the core.** Make `n_categories` a parameter; build `w`/`g` vectors programmatically; move the 64-channel / decimate / metadata assumptions into `frontends/eeg_mne.py`. Rename classes for PEP8/clarity while you're still `0.x`. **Deliverable: the core is genuinely modality-agnostic; the EEG pipeline still reproduces your SOUNDMODEL numbers.**

**Phase 4 — expose the dynamics.** Add `model.drift()` / `drift_deriv()`. Migrate the bifurcation + metastability logic from SOUNDMODEL into `lead/dynamics.py`. **Deliverable: the scientific analysis is a library call, not a notebook.**

**Phase 5 — tests + CI.** The four oracle tests above; GitHub Actions. **Deliverable: green badge, safe to refactor further.**

**Phase 6 — one worked example + minimal docs.** A single clean `examples/` notebook: EEG `.fif` → `LatentSeries` → fit linear + one bifurcation model → compare → bifurcation probability → plot. This *is* your tutorial and your integration test. **Deliverable: a colleague can copy it and swap in their data.**

Second pass (later): `compare.py` migration, additional frontends (iEEG, fMRI), a rendered doc site, PyPI upload, behavioral-prediction promotion.

---

## 8. Design questions decided before Phase 3 (2026-09-16)

**Resolved:** (1) keep discrete `category`, free only its cardinality; (2) keep a designated baseline category but expose it as `baseline_category=0`; (3) rename to public `*LEAD` names with deprecated aliases. Original framing kept below for the record.

These are the load-bearing choices where I'd want your call rather than guessing:

1. **Is "category" the right core abstraction, or should the core take a continuous input `I(t)` and let *frontends* bin into categories?** Your SNR levels are discrete, but iEEG/fMRI paradigms may have continuous or differently-structured drives. Keeping `category` in the core is simpler now; making the core continuous-input and treating categories as a frontend convenience is more general but a bigger change. Foundation-first argues: keep `category` but make its cardinality free (Phase 3), revisit continuity only if a real dataset demands it.
2. **How much does the core need to know about a "baseline/resting" category?** Several fitting strategies assume category 0 is resting-state with `w0=0`. Generalise to "an optional designated baseline category", or drop the assumption and let the frontend pass a flag.
3. **Naming for a public audience.** `Stratified*`, `LEAD_abstract`, `w`/`g` are insider names. Worth a short glossary decision now (e.g. `LinearLEAD`, `GainModulationLEAD`, `BaseLEADModel`) so the API reads well to a first-time user — cheap while `0.x`, expensive after.

---

*If it helps, the natural next step is Phase 0+1 as an actual PR (gitignore, src-layout, pyproject with the extras split, README) — that's mechanical and gets you an installable package to build the rest on. Say the word and I'll draft those files.*
