"""Per-modality frontends for LEADyna.

A frontend turns raw modality-specific recordings into the modality-agnostic
latent time-series the core consumes (a ``dict[category -> (n_trials,
n_timesteps)]`` or a :class:`~leadyna.datasets.LatentSeries`). EEG (MNE) is the
only frontend for now; iEEG / infant-EEG / fMRI come later as new modules here,
written against the same contract. Frontends pull in optional dependencies and
are imported lazily from the top-level package.
"""
