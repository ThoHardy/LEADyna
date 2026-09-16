"""Deprecated class-name aliases must keep resolving (to the new class) while
emitting a DeprecationWarning, so existing SOUNDMODEL notebooks keep running.
"""
import warnings

import pytest

from leadyna import model as m

_ALIASES = {
    "LEAD_abstract": "BaseLEADModel",
    "StratifiedLinear": "LinearLEAD",
    "NonLinear1": "SigmoidFeedbackLEAD",
    "StratifiedNonLinear1": "StratifiedSigmoidFeedbackLEAD",
    "NonLinear2": "AffineFeedbackLEAD",
    "StratifiedNonLinear2": "StratifiedAffineFeedbackLEAD",
    "GainModulation": "GainModulationLEAD",
    "StratifiedGainModulation": "StratifiedGainModulationLEAD",
}


@pytest.mark.parametrize("old,new", list(_ALIASES.items()))
def test_alias_warns_and_resolves(old, new):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = getattr(m, old)
    assert obj is getattr(m, new)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        m.DefinitelyNotAModel
