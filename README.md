# LEADyna

**Latent Evidence Accumulation Dynamics** — fit stochastic dynamical-systems models
(linear OU and sigmoid-bifurcation variants) to latent neural time-series, with a
UKF-based maximum-likelihood engine.

The library is organized around a modality-agnostic core (models + fitting) and
optional, swappable *frontends* that turn raw recordings into latent time-series.
The core consumes plain arrays and knows nothing about any recording modality; the
EEG/MNE frontend is the only one shipped so far.

## Install

```bash
# core only (models + fitting)
pip install leadyna

# with the EEG frontend (MNE + scikit-learn + pandas) and plotting
pip install "leadyna[eeg,viz]"

# development (tests, linter, notebook-output stripping)
pip install -e ".[eeg,viz,dev]"
```

Python 3.11+. Install into an isolated environment (venv or conda env), not your
base environment.

## Tutorial: from EEG epochs to model comparison

The workflow has four steps. Steps 1 and 3 are done by LEADyna; steps 0 and 2 are
yours, on purpose: they encode your experimental design and your input hypothesis.

| Step | Who | What |
|---|---|---|
| 0. Format the epochs | you | an MNE epochs object with `metadata["snr"]` (integer condition labels, baseline = 0) and a block column |
| 1. Decode the latent | `decode_latent` | cross-validated decoder → `LatentSeries` of `s(t)` per trial, category 0 = baseline |
| 2. State the input hypothesis | you | choose where `I(t)` is assumed constant, cut the windows, build `inputs` |
| 3. Fit and compare | `clever_fit_*` | fit Linear / Fixed gain / Modulated gain, compare held-out log-likelihoods |

The full script is in [`examples/toy_example.py`](examples/toy_example.py) (about 8 minutes on one core).

### Step 0 — epochs in the expected format

The frontend accepts **any MNE epochs object** (`mne.Epochs`, `mne.EpochsArray`,
epochs read from EEGLAB or FIF — anything deriving from `mne.BaseEpochs`) or a path to
a `-epo.fif` file. It needs:

- `epochs.metadata["snr"]`: an integer-like condition label per epoch. The baseline
  (resting state / noise only) can carry any label; you give it with `baseline_label`
  (default `0`). Other labels are the stimulus levels.
- a block column (default `"blocknumber"`) for leave-one-block-out decoding, or pass
  `cv=k` for stratified k-fold instead.

Here we simulate 13-channel EEG at 250 Hz: rest epochs plus four SNR levels, where a
late pattern switches on in an all-or-none way with a probability that grows with SNR.

```python
import mne
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
sfreq, n_channels, n_per_level = 250.0, 13, 20
times = np.arange(-50, 250) / sfreq                    # -0.2 s ... 0.996 s
levels = rng.permutation(np.repeat([0, 1, 2, 3, 4], n_per_level))  # 0 = rest, 1-4 = SNR
p_access = np.array([0.0, 0.1, 0.3, 0.6, 0.9])        # all-or-none "ignition" probability
accessed = rng.random(len(levels)) < p_access[levels]
pattern = rng.normal(size=n_channels)                  # topography of the late pattern
ignition = 1 / (1 + np.exp(-(times - 0.35) / 0.03))   # switches on around 350 ms

X = rng.normal(size=(len(levels), n_channels, len(times)))
X += 0.6 * accessed[:, None, None] * pattern[None, :, None] * ignition[None, None, :]
metadata = pd.DataFrame({
    "snr": levels,                                                  # integer labels, 0 = rest
    "blocknumber": np.arange(len(levels)) * 5 // len(levels) + 1,   # 5 contiguous blocks
})
info = mne.create_info(n_channels, sfreq, "eeg")
epochs = mne.EpochsArray(X * 1e-6, info, tmin=times[0], metadata=metadata)
```

With real data you would get `epochs` from your own loader instead — for example
`mne.Epochs(raw, events, tmin=-0.2, tmax=1.0, metadata=metadata, baseline=None)`,
`mne.read_epochs("sub01-epo.fif")`, or `mne.read_epochs_eeglab(...)` followed by
`mne.concatenate_epochs` (which returns an `EpochsArray`). All of them go through
step 1 identically, and an `mne.Epochs` and an `mne.EpochsArray` holding the same data
give the same latent (this is tested).

**About block numbers.** If your data have no blocks, you can assign them yourself
before this step (contiguous blocks in trial order, as above, or at random). Either is
safe for LEADyna: `decode_latent` returns the trials of each category **in their
original epoch order**, whatever the folds, and records each row's epoch index in
`latent.metadata["trial_index"][cat]`. Use that index — not an assumed relation
between trial order and block order — to match latent values with behaviour. (The old
LEAD `STG` returned trials sorted by block; analyses written for it assumed
increasing trial number = increasing block number.)

### Step 1 — decode the latent time-series

```python
from leadyna import decode_latent

latent = decode_latent(epochs, train_window=(0.4, 0.8))
latent.category_labels     # {0: 'snr=0 (baseline)', 1: 'snr=1', ..., 4: 'snr=4'}
latent.states[4].shape     # (20, 120): 20 trials x 120 samples at 100 Hz
latent.times               # sample times in seconds
```

A logistic-regression decoder (all channels) is trained to separate the baseline from
the strongest stimulus inside `train_window` (seconds), with leave-one-block-out
cross-validation, then applied to every sample of every held-out epoch. The result is
demeaned by the time-resolved baseline mean and scaled by the baseline standard
deviation, so the latent is in units of resting-state variability.

Conventions:

- **Category 0 is always the baseline.** The other labels are sorted and mapped to
  1, 2, ... (`latent.metadata["label_to_category"]` keeps the mapping).
- **The time grid is resampled to `target_sfreq` (default 100 Hz)**, whatever the
  recording rate: integer ratios are decimated, others resampled. One model step
  (`dt = 1`) is then always 10 ms, so fitted `tau` values and the default bounds mean
  the same thing across datasets. Pass `target_sfreq=None` to keep the native rate.
- `latent.inputs` is `None`: the frontend does not know your input hypothesis.

Check that the decoder actually decodes (e.g. accuracy or AUC through time per subject)
before going further: LEADyna does not do that for you, and a model fitted on a latent
that carries no stimulus information is meaningless.

### Step 2 — state your input hypothesis

The models are `dx = [-x/tau + w_c * I(t) + nonlinearity(x)] dt + noise`. You decide in
which window the input `I(t)` is assumed constant (here `I = 1` from 400 to 800 ms),
and which segments define the baseline dynamics (`I = 0`): the rest epochs, cut into
segments of the pre-stimulus length, plus the pre-stimulus part of every stimulus trial.
The fitting functions receive these windows **already cut**.

```python
t = np.round(latent.times, 6)
stim_window = (t >= 0.4) & (t < 0.8)   # I(t) = 1 here (assumed constant)
pre_stim = t < 0                       # no input before the stimulus
seg = int(pre_stim.sum())              # baseline segments have the pre-stimulus length

rest = latent.states[0]
rest_segments = [rest[:, k * seg:(k + 1) * seg] for k in range(rest.shape[1] // seg)]
prestim_segments = [latent.states[c][:, pre_stim] for c in latent.categories if c != 0]

states = {0: np.concatenate(rest_segments + prestim_segments)}
states |= {c: latent.states[c][:, stim_window] for c in latent.categories if c != 0}
inputs = {0: np.zeros_like(states[0])}
inputs |= {c: np.ones_like(x) for c, x in states.items() if c != 0}
```

Every category is one 2-D array `(n_sequences, n_samples)`; sequences of different
categories can have different lengths. Each sequence is filtered independently, so
cutting the baseline into segments is harmless.

### Step 3 — fit the three models and compare them

```python
from leadyna import LatentSeries
from leadyna.fitting_tools import clever_fit_gainmodul, clever_fit_linear, clever_fit_nonlinear1


def split(states, inputs, test_fraction=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_s, train_i, test_s, test_i = {}, {}, {}, {}
    for c, x in states.items():
        order = rng.permutation(len(x))
        n_test = int(round(test_fraction * len(x)))
        test, train = order[:n_test], order[n_test:]
        train_s[c], train_i[c] = x[train], inputs[c][train]
        test_s[c], test_i[c] = x[test], inputs[c][test]
    return LatentSeries(train_s, train_i), LatentSeries(test_s, test_i)


train, test = split(states, inputs)

light = dict(threshold_grid=(0.5, 1.5), n_loops=1)   # light search for the toy; drop for real data
linear = clever_fit_linear(train, n_jobs=1)
fixed_gain = clever_fit_nonlinear1(linear, train, n_jobs=1, **light)
modulated_gain = clever_fit_gainmodul(linear, train, n_jobs=1, **light)

for name, m in [("Linear", linear), ("Fixed gain", fixed_gain), ("Modulated gain", modulated_gain)]:
    print(f"{name:15s} held-out log-likelihood: {m.loglikelihood(test, n_jobs=1):9.1f}")
```

On the toy data this prints (about 8 minutes on one core):

```
Linear          held-out log-likelihood:   -2827.7
Fixed gain      held-out log-likelihood:   -2306.1
Modulated gain  held-out log-likelihood:   -2466.1
```

The fixed-gain model wins, as it should: in the simulation the "accessed" state has the
same amplitude at every SNR and only its probability changes, which is what a single
shared non-linearity produces. The light search (`threshold_grid=(0.5, 1.5)`,
`n_loops=1`) keeps the toy short; drop it for real analyses.

| Model | Class | Fitting function |
|---|---|---|
| Linear (OU) | `LinearLEAD` | `clever_fit_linear` |
| Fixed gain | `StratifiedSigmoidFeedbackLEAD` | `clever_fit_nonlinear1` |
| Modulated gain (one gain per category) | `StratifiedGainModulationLEAD` | `clever_fit_gainmodul` |

`clever_fit_null` (one OU process, all weights 0) and `clever_fit_nonlinear2` (affine
feedback) are also available. For a real analysis, replace the single split by k-fold
cross-validation (split each category into folds, fit on k-1, sum the held-out
log-likelihoods) and compare models across subjects with group-level Bayesian model
selection.

Fitting options shared by the `clever_fit_*` functions:

- `bounds`: override any default bound, e.g. `bounds={"tau": (1, 60)}` for slower
  dynamics. Defaults (`leadyna.fitting_tools.DEFAULT_BOUNDS`) are the SOUNDMODEL values:
  `tau` (1, 25), noises (0.01, 1), weights `w` and gains `g` (0, 1), fixed gain
  (0, 0.5), threshold (0, 2), sharpness (0, 10). Bounds are in units of `dt`.
- `threshold_grid` and `sharpness` (non-linear models): initial thresholds tried and the
  fixed sigmoid sharpness (defaults `linspace(0, 2, 5)` and 5).
- `n_jobs`: `1` for small or local fits; leave unset (8 workers) for large datasets.

Missing stimulus categories are fine (a subject with no trial at one level simply has
no key for it); category 0 is required.

## The core contract: `LatentSeries`

The core consumes latent time-series in one validated format — a `LatentSeries`
(states / inputs / `dt` / category labels / metadata). Supporting a new modality
(iEEG, fMRI) means writing a frontend that returns one of these; the whole downstream
stack (fit, likelihood) then works unchanged.

```python
from leadyna import LatentSeries, LinearLEAD

states = {0: np.random.randn(20, 50), 1: np.random.randn(20, 50)}
inputs = {0: np.zeros((20, 50)),       1: np.ones((20, 50))}
data = LatentSeries(states, inputs, dt=1.0, category_labels={0: "rest", 1: "stim"})

model = LinearLEAD(tau=10.0, process_noise=0.2, measure_noise=0.2, n_categories=2, w1=0.5)
model.fit(data,
          init_params=[10, 0.2, 0.2] + [0] * 2,
          bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)] * 2,
          fixed_params=["tau", "process_noise", "measure_noise", "w0"],
          n_jobs=1)
```

`inputs` may be `None` (a frontend's output); attach them with
`latent.with_inputs(inputs)` before fitting. `dt` defaults to `1.0` (one sample = one
step), which reproduces the historical numerics; with the EEG frontend's default
`target_sfreq=100`, one step is 10 ms. `fit` and `loglikelihood` also accept the legacy
`(state_series, input_series)` dict pair.

## Migrating from LEAD

- `clever_fit_*` now take **already-windowed** data. In LEAD they cut the stimulus
  window themselves with `input_start_index=75, input_stop_index=100` as defaults.
  Passing these two arguments still reproduces the LEAD behaviour (with a
  `DeprecationWarning`); results are bit-identical to LEAD (tested, `pytest -m slow`).
- `STG(path, tmin, tmax)` (milliseconds, Sergent et al. 2021 conventions, dict output)
  still works but is deprecated in favour of `decode_latent` (seconds, `LatentSeries`
  output). The experimental `substract_pattern` and `sort_by_snr=False` options were
  removed.
- Model classes have public names (`LinearLEAD`, `StratifiedSigmoidFeedbackLEAD`,
  `StratifiedGainModulationLEAD`, ...); the old names still resolve with a
  `DeprecationWarning`.

## Status

Alpha (`0.x`): the public API may change between minor versions.

## License

MIT — see [LICENSE](LICENSE).
