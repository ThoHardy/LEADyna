"""EEG (MNE) frontend: from epoched EEG to a :class:`~leadyna.datasets.LatentSeries`.

The latent variable ``s(t)`` is the cross-validated decision value of a linear
decoder (logistic regression on all channels) trained to separate *baseline*
(resting / noise-only) epochs from the *strongest-stimulus* epochs within a
training time window, then applied to every timepoint of every held-out epoch
(Segmented Temporal Generalization, as in SOUNDMODEL).

Conventions fixed by this frontend
----------------------------------
* **Category 0 is always the baseline** (resting state / noise only). Whatever
  label the baseline carries in your metadata (``baseline_label``) is mapped to
  category 0; the other labels are sorted ascending and mapped to 1, 2, ...
* **Trials keep their original epoch order** within each category, whatever the
  cross-validation folds. ``latent.metadata["trial_index"][cat]`` gives, for
  each row of ``latent.states[cat]``, its index in the input ``epochs``, so
  latent values can be matched to behaviour trial by trial even when block
  numbers were assigned at random.
* **The time grid is resampled to ``target_sfreq``** (default 100 Hz, i.e. one
  step = 10 ms), so model parameters expressed in index units (``dt=1``) mean
  the same thing across datasets recorded at different sampling rates.
"""
from __future__ import annotations

import warnings
from os import PathLike

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..datasets import LatentSeries

__all__ = ["decode_latent", "STG"]

BASELINE_CATEGORY = 0


def _as_epochs(epochs) -> mne.BaseEpochs:
    if isinstance(epochs, mne.BaseEpochs):
        return epochs.copy().load_data()
    if isinstance(epochs, (str, PathLike)):
        return mne.read_epochs(epochs, preload=True, verbose=False)
    raise TypeError(
        "epochs must be an mne.Epochs / mne.EpochsArray (any mne.BaseEpochs) or a path "
        f"to a -epo.fif file; got {type(epochs).__name__}."
    )


def _resample(epochs: mne.BaseEpochs, target_sfreq) -> mne.BaseEpochs:
    """Bring ``epochs`` to ``target_sfreq``: decimate on integer ratios, resample otherwise.

    Integer ratios use ``decimate`` (pure subsampling, as in LEAD/SOUNDMODEL, so the
    historical numerics are reproduced exactly); non-integer ratios fall back to
    MNE's FFT-based ``resample``.
    """
    if target_sfreq is None:
        return epochs
    sfreq = float(epochs.info["sfreq"])
    if target_sfreq > sfreq:
        raise ValueError(
            f"target_sfreq={target_sfreq} Hz exceeds the data sampling rate ({sfreq} Hz)."
        )
    ratio = sfreq / target_sfreq
    if np.isclose(ratio, round(ratio)):
        if round(ratio) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                epochs.decimate(int(round(ratio)), verbose=False)
    else:
        epochs.resample(target_sfreq, verbose=False)
    return epochs


def _labels(epochs: mne.BaseEpochs, condition: str) -> np.ndarray:
    if epochs.metadata is None or condition not in epochs.metadata.columns:
        raise ValueError(
            f"epochs.metadata must have a '{condition}' column giving each epoch's "
            "condition (baseline / stimulus level)."
        )
    raw = epochs.metadata[condition].to_numpy()
    try:
        as_float = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata['{condition}'] must hold integer-like labels (e.g. 0, 1, 2 or "
            f"'0', '1', '2'); got values like {raw[:3].tolist()}."
        ) from exc
    if not np.all(np.isfinite(as_float)) or not np.allclose(as_float, np.round(as_float)):
        raise ValueError(f"metadata['{condition}'] must hold integer-like labels.")
    return np.round(as_float).astype(int)


def _fold_ids(epochs, labels, cv, random_state) -> np.ndarray:
    if isinstance(cv, str):
        if cv not in epochs.metadata.columns:
            raise ValueError(
                f"cv='{cv}' names a metadata column that does not exist. Add a block "
                "column to epochs.metadata, or pass an integer cv for stratified K-fold."
            )
        return epochs.metadata[cv].to_numpy()
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool) and cv >= 2:
        folds = np.empty(len(labels), dtype=int)
        skf = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state)
        for k, (_, test) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            folds[test] = k
        return folds
    raise ValueError("cv must be a metadata column name (str) or an integer >= 2.")


def decode_latent(
    epochs,
    train_window,
    *,
    condition: str = "snr",
    baseline_label=0,
    present_label=None,
    cv="blocknumber",
    target_sfreq: float | None = 100.0,
    random_state: int = 0,
) -> LatentSeries:
    """Extract a latent time-series per trial from epoched EEG.

    Parameters
    ----------
    epochs : mne.BaseEpochs or path-like
        Any MNE epochs object (``mne.Epochs``, ``mne.EpochsArray``, epochs read
        from EEGLAB/FIF...) or a path to a ``-epo.fif`` file. Must carry
        ``metadata`` with a ``condition`` column (and a block column if ``cv`` is
        a column name). The input object is never modified.
    train_window : tuple of float
        ``(tmin, tmax)`` in **seconds**: window in which the decoder is trained
        (e.g. ``(0.3, 0.5)`` for a P3-like late pattern).
    condition : str
        Metadata column holding integer-like condition labels (default ``"snr"``).
    baseline_label : int
        Label of the baseline (resting / noise-only) epochs; mapped to category 0.
    present_label : int, optional
        Label of the epochs the decoder learns to separate from baseline. Defaults
        to the largest label (strongest stimulus).
    cv : str or int
        Decoder cross-validation. A metadata column name (default
        ``"blocknumber"``) gives leave-one-block-out; an integer ``k`` gives
        stratified ``k``-fold over epochs. Every epoch's latent comes from a
        decoder that never saw it.
    target_sfreq : float or None
        Sampling rate (Hz) of the output time grid; ``None`` keeps the native rate.
    random_state : int
        Seed for the stratified folds when ``cv`` is an integer.

    Returns
    -------
    LatentSeries
        ``states[cat]`` of shape ``(n_trials_cat, n_times)``, demeaned by the
        (time-resolved) baseline mean and scaled by the baseline standard
        deviation; ``inputs=None`` (the input hypothesis is yours to define);
        ``dt=1.0`` (one step = ``1 / target_sfreq`` s). ``metadata`` holds
        ``times``, ``sfreq``, ``trial_index``, ``label_to_category`` and the
        decoding settings.
    """
    ep = _resample(_as_epochs(epochs), target_sfreq)
    labels = _labels(ep, condition)
    present_values = np.unique(labels)

    if baseline_label not in present_values:
        raise ValueError(
            f"baseline_label={baseline_label!r} not found in metadata['{condition}'] "
            f"(labels present: {present_values.tolist()})."
        )
    stim_values = [v for v in present_values if v != baseline_label]
    if not stim_values:
        raise ValueError("No stimulus epochs: every epoch carries the baseline label.")
    if present_label is None:
        present_label = max(stim_values)
    if present_label not in stim_values:
        raise ValueError(f"present_label={present_label!r} not found among {stim_values}.")

    label_to_cat = {int(baseline_label): BASELINE_CATEGORY}
    label_to_cat |= {int(v): k + 1 for k, v in enumerate(sorted(stim_values))}

    tmin, tmax = train_window
    train_ep = ep.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True, verbose=False)
    X_window = train_ep.get_data()
    X_full = ep.get_data()
    n_epochs, n_channels, n_times = X_full.shape

    folds = _fold_ids(ep, labels, cv, random_state)
    is_base, is_present = labels == baseline_label, labels == present_label
    latent = np.full((n_epochs, n_times), np.nan)

    for fold in np.unique(folds):
        test = folds == fold
        train_idx = np.concatenate([
            np.flatnonzero(~test & is_base),
            np.flatnonzero(~test & is_present),
        ])
        if not (np.any(is_base[train_idx]) and np.any(is_present[train_idx])):
            raise ValueError(
                f"Fold '{fold}' leaves no baseline or no '{present_label}' epoch for "
                "training; use fewer / larger folds."
            )
        X_train = X_window[train_idx].transpose(0, 2, 1).reshape(-1, n_channels)
        y_train = np.repeat(is_present[train_idx].astype(int), X_window.shape[2])
        decoder = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("classifier", LogisticRegression(solver="liblinear")),  # L2 penalty (default)
        ])
        decoder.fit(X_train, y_train)

        X_test = X_full[test].transpose(0, 2, 1).reshape(-1, n_channels)
        latent[test] = decoder.decision_function(X_test).reshape(-1, n_times)

    base = latent[is_base]
    centered = latent - base.mean(axis=0)
    scale = np.std(centered[is_base])

    states, trial_index = {}, {}
    for label, cat in label_to_cat.items():
        idx = np.flatnonzero(labels == label)
        states[cat] = centered[idx] / scale
        trial_index[cat] = idx

    return LatentSeries(
        states=states,
        inputs=None,
        dt=1.0,
        category_labels={cat: f"{condition}={label}" + (" (baseline)" if cat == 0 else "")
                         for label, cat in label_to_cat.items()},
        metadata={
            "times": ep.times.copy(),
            "sfreq": float(ep.info["sfreq"]),
            "trial_index": trial_index,
            "label_to_category": label_to_cat,
            "train_window": (tmin, tmax),
            "present_label": int(present_label),
            "cv": cv,
            "n_channels": n_channels,
            "frontend": "eeg_mne.decode_latent",
        },
    )


def STG(data_ref, tmin, tmax, decimate_factor=5, **removed):
    """Deprecated LEAD-era entry point; use :func:`decode_latent`.

    Kept so SOUNDMODEL notebooks run unchanged: windows in **milliseconds**,
    baseline label ``1`` (Sergent et al. 2021 catch trials), leave-one-block-out
    on ``blocknumber``, and a plain ``dict[category -> array]`` return value.
    ``data_ref`` may now also be an in-memory ``mne.BaseEpochs``.
    The experimental ``substract_pattern`` and ``sort_by_snr=False`` options were
    removed from LEADyna.
    """
    if removed:
        raise TypeError(
            f"STG no longer supports {sorted(removed)} (removed in LEADyna); "
            "use decode_latent."
        )
    warnings.warn(
        "STG is deprecated; use leadyna.frontends.eeg_mne.decode_latent "
        "(windows in seconds, returns a LatentSeries).",
        DeprecationWarning,
        stacklevel=2,
    )
    ep = _as_epochs(data_ref)
    latent = decode_latent(
        ep,
        (tmin / 1000, tmax / 1000),
        baseline_label=1,
        cv="blocknumber",
        target_sfreq=ep.info["sfreq"] / decimate_factor,
    )
    return latent.states
