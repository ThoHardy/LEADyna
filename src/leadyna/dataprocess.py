"""Deprecated module path. The EEG (MNE) frontend moved to
``leadyna.frontends.eeg_mne`` in Phase 3; import from there instead.

This shim re-exports the public API and will be removed in a future release.
"""
import warnings as _warnings

from .frontends.eeg_mne import STG  # noqa: F401

_warnings.warn(
    "leadyna.dataprocess has moved to leadyna.frontends.eeg_mne; "
    "update your imports (e.g. `from leadyna.frontends.eeg_mne import STG`).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["STG"]
