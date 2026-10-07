"""LEADyna toy example: simulated EEG epochs -> latent time-series -> 3 models -> comparison.

Run with:  python examples/toy_example.py
Same code as the README tutorial. Runs in under a minute.
"""
import mne
import numpy as np
import pandas as pd

from leadyna import LatentSeries, decode_latent
from leadyna.fitting_tools import clever_fit_gainmodul, clever_fit_linear, clever_fit_nonlinear1

mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
# 0. Toy EEG epochs in the format the frontend expects
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
sfreq, n_channels, n_per_level = 250.0, 13, 40
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

# ---------------------------------------------------------------------------
# 1. Frontend: decode a latent time-series s(t) for every trial
# ---------------------------------------------------------------------------
latent = decode_latent(epochs, train_window=(0.4, 0.8))   # resampled to 100 Hz (dt = 10 ms)
print(latent.category_labels)
print({c: latent.states[c].shape for c in latent.categories})

# ---------------------------------------------------------------------------
# 2. Your input hypothesis I(t) and the windows it applies to
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Fit the three models on 80 % of the trials, compare on the other 20 %
# ---------------------------------------------------------------------------
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

linear = clever_fit_linear(train)
fixed_gain = clever_fit_nonlinear1(linear, train)
modulated_gain = clever_fit_gainmodul(linear, train)

for name, m in [("Linear", linear), ("Fixed gain", fixed_gain), ("Modulated gain", modulated_gain)]:
    print(f"{name:15s} held-out log-likelihood: {m.loglikelihood(test):9.1f}")
