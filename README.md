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

# with the EEG frontend (MNE + scikit-learn) and plotting
pip install "leadyna[eeg,viz]"

# development (tests, linter, notebook-output stripping)
pip install -e ".[eeg,viz,dev]"
```

Python 3.11+.

## Quickstart

```python
import numpy as np
import leadyna
from leadyna import STG                          # EEG frontend (needs the [eeg] extra)
from leadyna.fitting_tools import clever_fit_linear, clever_fit_gainmodul

# 1. latent time-series from EEG epochs (dict: category -> (n_trials, n_timesteps))
states = STG("subject01_epochs.fif", tmin=250, tmax=500)
inputs = {cat: np.ones_like(x) for cat, x in states.items()}

# 2. linear baseline, then a bifurcation model warm-started from it
linear = clever_fit_linear(states, inputs)
gain = clever_fit_gainmodul(linear, states, inputs)

print("linear  LL:", linear.loglikelihood(states, inputs))
print("gainmod LL:", gain.loglikelihood(states, inputs))
```

## The core contract: `LatentSeries`

The core consumes latent time-series in one validated format — a `LatentSeries`
(states / inputs / `dt` / category labels). Supporting a new modality (iEEG,
infant-EEG, fMRI) means writing a frontend that returns one of these; the whole
downstream stack (fit, likelihood) then works unchanged.

```python
import numpy as np
from leadyna import LatentSeries
from leadyna.model import StratifiedLinear

# states / inputs: dict[category -> (n_trials, n_timesteps)]
states = {0: np.random.randn(20, 50), 1: np.random.randn(20, 50)}
inputs = {0: np.zeros((20, 50)),       1: np.ones((20, 50))}

data = LatentSeries(states, inputs, dt=1.0,           # dt=1.0 -> index units
                    category_labels={0: "rest", 1: "stim"})

model = StratifiedLinear(tau=10.0, process_noise=0.2, measure_noise=0.2, w1=0.5)
model.fit(data,
          init_params=[10, 0.2, 0.2] + [0] * 7,
          bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)] * 7,
          fixed_params=["tau", "process_noise", "measure_noise"]
                       + [f"w{k}" for k in range(7) if k != 1])
print("fitted w1:", model.w1, "| log-likelihood:", model.loglikelihood(data))
```

`dt` is required but defaults to `1.0` (one sample = one step), which reproduces
the library's historical numerics exactly. Set it to your physical timestep (in
seconds) when you want `tau` / noise in physical units — that changes their
meaning, so fits are only comparable at a fixed `dt`. `fit` and `loglikelihood`
still accept the legacy `(state_series, input_series)` dict pair, so existing
code keeps working.

### Fitting performance (local runs)

The UKF likelihood parallelises over trials with `joblib`. The engine still
defaults to `n_jobs=8`, which is worth it on real datasets but is pure overhead
on small data — on a handful of trials, spawning workers can make a fit *slower*
by an order of magnitude and oversubscribes a machine with fewer than 8 cores.
For small or local fits, pass `n_jobs=1`:

```python
model.fit(data, init_params=..., bounds=..., n_jobs=1)   # no worker spawn
```

`fit(...)` forwards `n_jobs` and `batch_size` to the likelihood; leaving them
unset keeps the historical `n_jobs=8` default.

## Status

Alpha (`0.x`): the public API may change between minor versions.

## License

MIT — see [LICENSE](LICENSE).
