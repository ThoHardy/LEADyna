from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from . import datasets, fitting_tools, model
from .datasets import LatentSeries

try:
    __version__ = _pkg_version("leadyna")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Frontends pull in optional dependencies (mne/scikit-learn for EEG, matplotlib/seaborn
# for plotting). They are imported lazily so the core installs and imports on its own.
_FRONTEND_EXTRA = {"dataprocess": "eeg", "visual": "viz"}
_FRONTEND_ATTR = {"STG": "dataprocess", "colormap": "visual"}


def __getattr__(name):
    if name in _FRONTEND_EXTRA:
        try:
            mod = import_module(f".{name}", __name__)
        except ImportError as exc:
            extra = _FRONTEND_EXTRA[name]
            raise ImportError(
                f"leadyna.{name} needs the optional '{extra}' dependencies. "
                f"Install them with: pip install 'leadyna[{extra}]'"
            ) from exc
        globals()[name] = mod
        return mod
    if name in _FRONTEND_ATTR:
        return getattr(__getattr__(_FRONTEND_ATTR[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatentSeries",
    "STG",
    "colormap",
    "datasets",
    "dataprocess",
    "fitting_tools",
    "model",
    "visual",
    "__version__",
]
