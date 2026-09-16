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

## Status

Alpha (`0.x`): the public API may change between minor versions.

## License

MIT — see [LICENSE](LICENSE).
