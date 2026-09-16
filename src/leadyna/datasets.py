"""The LatentSeries contract — the boundary between LEADyna's modality-agnostic
core and any per-modality frontend.

A frontend's only job is to turn raw recordings into a :class:`LatentSeries`;
the core (models, UKF likelihood, fitting) consumes nothing else. Validating the
format in one place is what makes "support iEEG / infant-EEG / fMRI" reduce to
"write a frontend that returns a LatentSeries".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LatentSeries"]

_NUMBER = (int, float, np.integer, np.floating)


@dataclass
class LatentSeries:
    """A validated container of latent time-series, keyed by category.

    Parameters
    ----------
    states : dict[int, numpy.ndarray]
        Latent variable ``s(t)`` per category, each array of shape
        ``(n_trials, n_timesteps)``.
    inputs : dict[int, numpy.ndarray]
        Exogenous drive ``I(t)`` per category, same shape as the matching
        ``states`` entry.
    dt : float, optional
        Timestep between successive samples. Defaults to ``1.0`` (index units:
        one sample = one step), which reproduces the library's historical
        behaviour exactly. Set it to the physical timestep (in seconds) when you
        want ``tau`` / noise expressed in physical units — this changes their
        numerical meaning, so fitted parameters are only comparable at a fixed
        ``dt``.
    category_labels : dict[int, str], optional
        Human-readable meaning of each category (e.g. ``{0: "rest", 1: "SNR1"}``)
        for plots and reports. Keys must be a subset of ``states``.
    metadata : dict, optional
        Free-form provenance (subject id, modality, sampling rate…). Never read
        by the core.

    Notes
    -----
    The UKF likelihood is a product over timesteps, so at least two timesteps are
    required. The core runs on any ``>= 2``-timestep series; statistical power is
    a property of the data, not the library.
    """

    states: dict
    inputs: dict
    dt: float = 1.0
    category_labels: dict | None = None
    metadata: dict | None = None

    def __post_init__(self):
        self.states = {k: np.asarray(v, dtype=float) for k, v in dict(self.states).items()}
        self.inputs = {k: np.asarray(v, dtype=float) for k, v in dict(self.inputs).items()}
        self.validate()

    # -- introspection ------------------------------------------------------
    @property
    def categories(self) -> list:
        return list(self.states.keys())

    def n_trials(self, category) -> int:
        return int(self.states[category].shape[0])

    def n_timesteps(self, category) -> int:
        return int(self.states[category].shape[1])

    def __len__(self) -> int:
        return len(self.states)

    # -- validation ---------------------------------------------------------
    def validate(self) -> LatentSeries:
        """Check the contract; raise a clear error on the first violation."""
        if isinstance(self.dt, bool) or not isinstance(self.dt, _NUMBER):
            raise TypeError(f"dt must be a real number, got {type(self.dt).__name__}.")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"dt must be finite and > 0, got {self.dt!r}.")

        if not self.states:
            raise ValueError("states is empty: at least one category is required.")

        if set(self.states) != set(self.inputs):
            raise ValueError(
                "states and inputs must share identical category keys; "
                f"states has {sorted(self.states)}, inputs has {sorted(self.inputs)}."
            )

        for cat, arr in self.states.items():
            inp = self.inputs[cat]
            if arr.ndim != 2:
                raise ValueError(
                    f"states[{cat!r}] must be 2-D (n_trials, n_timesteps); got shape "
                    f"{arr.shape}. Reshape a single trial with arr[None, :]."
                )
            if inp.ndim != 2:
                raise ValueError(
                    f"inputs[{cat!r}] must be 2-D (n_trials, n_timesteps); got shape {inp.shape}."
                )
            if arr.shape != inp.shape:
                raise ValueError(
                    f"states[{cat!r}] and inputs[{cat!r}] shapes differ: "
                    f"{arr.shape} vs {inp.shape}."
                )
            if arr.shape[1] < 2:
                raise ValueError(
                    f"category {cat!r} has {arr.shape[1]} timestep(s); the UKF likelihood "
                    "needs at least 2."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"states[{cat!r}] contains non-finite values.")
            if not np.all(np.isfinite(inp)):
                raise ValueError(f"inputs[{cat!r}] contains non-finite values.")

        if self.category_labels is not None:
            extra = set(self.category_labels) - set(self.states)
            if extra:
                raise ValueError(
                    f"category_labels has keys absent from states: {sorted(extra)}."
                )
        return self
