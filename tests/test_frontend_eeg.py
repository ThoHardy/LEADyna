"""EEG frontend: equivalence with the LEAD-era STG, conventions, and robustness."""
import warnings

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("sklearn")

from _legacy_stg import legacy_STG  # noqa: E402
from leadyna import LatentSeries  # noqa: E402
from leadyna.frontends.eeg_mne import STG, decode_latent  # noqa: E402

mne.set_log_level("ERROR")


def _toy(labels, sfreq, tmin, tmax, blocks, seed=0, n_ch=13, str_labels=False):
    rng = np.random.default_rng(seed)
    times = np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq))) / sfreq
    pattern = rng.normal(size=n_ch)
    top = max(labels)
    X = rng.normal(size=(len(labels), n_ch, len(times)))
    for i, lab in enumerate(labels):
        amp = (lab - min(labels)) / (top - min(labels))
        X[i] += 1.5 * amp * np.outer(pattern, 1 / (1 + np.exp(-(times - 0.3) / 0.03)))
    info = mne.create_info(n_ch, sfreq, "eeg")
    events = np.c_[np.arange(len(labels)) * 10_000, np.zeros(len(labels), int), labels]
    meta = pd.DataFrame({"snr": [str(v) for v in labels] if str_labels else labels,
                         "blocknumber": blocks})
    return mne.EpochsArray(X * 1e-6, info, events=events, tmin=times[0],
                           event_id={str(v): int(v) for v in np.unique(labels)},
                           metadata=meta, verbose=False)


@pytest.fixture(scope="module")
def sergent_like():
    rng = np.random.default_rng(1)
    labels = rng.permutation(np.repeat(np.arange(1, 8), 20))
    blocks = np.arange(len(labels)) * 5 // len(labels) + 1
    return _toy(labels, sfreq=500.0, tmin=-0.5, tmax=1.0, blocks=blocks)


@pytest.fixture(scope="module")
def baby_like():
    rng = np.random.default_rng(2)
    labels = rng.permutation(np.repeat(np.arange(0, 5), 30))
    blocks = rng.integers(1, 6, size=len(labels))  # random block assignment
    return _toy(labels, sfreq=250.0, tmin=-0.2, tmax=1.0, blocks=blocks, str_labels=True)


def test_legacy_stg_equivalence(sergent_like, tmp_path):
    fname = tmp_path / "sub-epo.fif"
    sergent_like.save(fname, fmt="double", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = legacy_STG(fname, tmin=300, tmax=500)
        new_path = STG(fname, tmin=300, tmax=500)
        new_obj = STG(sergent_like, tmin=300, tmax=500)
    assert sorted(old) == sorted(new_path) == list(range(7))
    for cat in old:
        np.testing.assert_allclose(new_path[cat], old[cat], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(new_obj[cat], old[cat], rtol=1e-9, atol=1e-9)


def test_stg_rejects_removed_options(sergent_like):
    with pytest.raises(TypeError, match="substract_pattern"):
        STG(sergent_like, 300, 500, substract_pattern=(0, 10))


def test_baseline_is_category_zero_and_labels_cast(baby_like):
    latent = decode_latent(baby_like, (0.3, 0.6))
    assert isinstance(latent, LatentSeries)
    assert latent.inputs is None
    assert sorted(latent.states) == [0, 1, 2, 3, 4]
    assert latent.metadata["label_to_category"] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    base = latent.states[0]
    assert np.allclose(base.mean(axis=0), 0, atol=1e-10)
    assert np.isclose(base.std(), 1.0)
    assert latent.states[4][:, latent.times > 0.4].mean() > 1.0


def test_baseline_label_remapped():
    rng = np.random.default_rng(3)
    labels = rng.permutation(np.repeat([5, 7, 9], 25))
    ep = _toy(labels, 200.0, -0.2, 0.8, blocks=np.tile(np.arange(5), 15))
    latent = decode_latent(ep, (0.3, 0.6), baseline_label=7)
    assert latent.metadata["label_to_category"] == {7: 0, 5: 1, 9: 2}
    assert latent.metadata["present_label"] == 9


def test_trial_order_preserved_with_random_blocks(baby_like):
    latent = decode_latent(baby_like, (0.3, 0.6))
    labels = baby_like.metadata["snr"].astype(int).to_numpy()
    for cat, idx in latent.metadata["trial_index"].items():
        assert np.all(np.diff(idx) > 0)
        assert np.all(labels[idx] == cat)
        assert len(idx) == latent.states[cat].shape[0]


def test_target_sfreq_non_integer_ratio(baby_like):
    latent = decode_latent(baby_like, (0.3, 0.6), target_sfreq=100.0)
    assert latent.metadata["sfreq"] == 100.0
    assert np.allclose(np.diff(latent.times), 0.01)
    native = decode_latent(baby_like, (0.3, 0.6), target_sfreq=None)
    assert native.metadata["sfreq"] == 250.0


def test_epochs_and_epochsarray_agree(baby_like):
    X = baby_like.get_data()
    n, n_ch, n_t = X.shape
    pad = 50
    cont = np.zeros((n_ch, n * (n_t + pad)))
    onsets = []
    for i in range(n):
        start = i * (n_t + pad)
        cont[:, start:start + n_t] = X[i]
        onsets.append(start - int(round(baby_like.tmin * baby_like.info["sfreq"])))
    raw = mne.io.RawArray(cont, baby_like.info, verbose=False)
    events = np.c_[onsets, np.zeros(n, int), baby_like.events[:, 2]]
    epochs = mne.Epochs(raw, events, tmin=baby_like.tmin, tmax=baby_like.times[-1],
                        baseline=None, metadata=baby_like.metadata, preload=True,
                        verbose=False)
    a = decode_latent(epochs, (0.3, 0.6))
    b = decode_latent(baby_like, (0.3, 0.6))
    for cat in a.states:
        np.testing.assert_allclose(a.states[cat], b.states[cat], atol=1e-10)


def test_integer_cv_without_blocks(baby_like):
    ep = baby_like.copy()
    ep.metadata = ep.metadata.drop(columns="blocknumber")
    with pytest.raises(ValueError, match="blocknumber"):
        decode_latent(ep, (0.3, 0.6))
    latent = decode_latent(ep, (0.3, 0.6), cv=5)
    assert sum(latent.n_trials(c) for c in latent.categories) == len(ep)


def test_input_not_modified(baby_like):
    before = baby_like.get_data().copy()
    decode_latent(baby_like, (0.3, 0.6))
    assert baby_like.info["sfreq"] == 250.0
    np.testing.assert_array_equal(baby_like.get_data(), before)


def test_bad_labels_and_baseline(baby_like):
    with pytest.raises(ValueError, match="baseline_label"):
        decode_latent(baby_like, (0.3, 0.6), baseline_label=9)
    ep = baby_like.copy()
    ep.metadata["snr"] = "rest"
    with pytest.raises(ValueError, match="integer-like"):
        decode_latent(ep, (0.3, 0.6))


def test_fit_requires_inputs(baby_like):
    from leadyna import LinearLEAD
    latent = decode_latent(baby_like, (0.3, 0.6))
    with pytest.raises(ValueError, match="with_inputs"):
        LinearLEAD(10, 0.1, 0.1, n_categories=5).loglikelihood(latent)
