"""Coupled stages: filter equivalence with the fast engine, pattern finding, K rules, recovery."""
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

from leadyna import coupled as cl  # noqa: E402
from leadyna.model import compute_ukf_loglikelihood_fast  # noqa: E402

EARLY = dict(tau=5.0, process_noise=0.25, onset=0.04, scale=0.03, amp=0.9)
LATE = dict(tau=10.0, process_noise=0.12, coupling=0.12, delay=0.08, gain=(0.25, 0.3, 0.0),
            threshold=1.5)
ANTI = [[1, -0.7], [-0.7, 1]]


@pytest.fixture(scope="module")
def two_stages():
    return cl.simulate_chain([EARLY, LATE], ANTI, seed=0)


def test_core_import_does_not_load_jax():
    code = "import sys, leadyna; assert not {'jax', 'leadyna.coupled'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("gain", [False, True])
def test_stage_ukf_matches_fast_engine(gain):
    rng = np.random.default_rng(0)
    y, parent = rng.normal(size=(40, 60)), rng.normal(size=(40, 60))
    tau, q, r, c, delay, g0, g1, g2, thr = 8.0, 0.3, 0.5, 0.4, 3.0, 0.2, 0.3, 0.1, 1.0
    stage = cl._Stage(first=False, gain=gain)
    theta = [tau, q, r, c, delay] + ([g0, g1, g2, thr] if gain else [])
    ll = cl._loglik(dict(stage=stage, theta=np.array(theta)), dict(y=y, parent=parent))

    u = np.zeros_like(parent)
    u[:, 3:] = parent[:, :-3]

    def fx(x, dt, u):
        g = np.maximum(0.0, g0 + g1 * u + g2 * u**2) if gain else 0.0
        return x + (-x / tau + c * u + g / (1 + np.exp(-cl.SHARPNESS * (x - thr))))

    a = 1 - 1 / tau
    ref = compute_ukf_loglikelihood_fast({0: y}, {0: u}, 1.0, q, r, lambda cat: fx,
                                         initial_moments={0: (0.0, q**2 / (1 - a**2))})
    assert ll == pytest.approx(ref, abs=1e-8)


def test_find_patterns_counts_two_stages(two_stages):
    pat = cl.find_patterns(two_stages["X"], two_stages["cats"], two_stages["times"], n_perm=10)
    assert pat["K_perm"] == 2 and pat["significant"] == [True, True, False, False]
    assert pat["B"].shape == (32, 4)
    assert np.isnan(pat["thr"][3])                 # no permutations after the first failure
    Z = cl.unmix(two_stages["X"], pat["B"], pat["Sigma"])
    assert Z.shape == (500, 4, 150)


def test_no_stage_in_pure_noise():
    sim = cl.simulate_chain([EARLY], [[1.0]], pattern_amp=0.0, n_per_level=60, seed=1)
    with pytest.raises(ValueError, match="K_perm = 0"):
        cl.fit_coupled(sim["X"], sim["cats"], sim["times"], n_perm=10)


def test_select_K_rules():
    res = dict(Ks=(1, 2, 3), K_perm=[2, 2, 3], null_ll=[-1000.0, -1010.0, -990.0],
               ll={1: [-700.0, -720.0, -690.0], 2: [-400.0, -430.0, -380.0],
                   3: [-395.0, -428.0, -377.0]})
    sel = cl.select_K(res)
    assert (sel["K_perm"], sel["K_elbow"], sel["K_ll"]) == (2, 2, 3)   # max LL over-selects
    res["ll"][1] = [-700.0, -1020.0, -690.0]                           # K=1 loses in one fold
    assert cl.select_K(res)["K_elbow"] == 0


@pytest.mark.slow
def test_two_stage_recovery(two_stages):
    model = cl.CoupledLEAD(n_perm=10).fit(two_stages["X"], two_stages["cats"], two_stages["times"])
    early, late = model.summary()
    assert model.K_ == 2
    assert (early["model"], late["model"]) == ("linear", "modulated_gain")
    assert late["parent"] == early["component"]
    assert abs(early["onset_s"] - 0.04) < 0.02 and abs(late["delay_s"] - 0.08) < 0.03
    # anticorrelated topographies are recovered; the late stage, found first by the
    # deflation, is cleaned of the sign-flipped copy of the early one (alpha < 0)
    r = np.abs(np.corrcoef(two_stages["A"].T, model.A_.T)[:2, 2:])
    assert r[0, early["component"]] > 0.8 and r[1, late["component"]] > 0.85
    assert late["component"] < early["component"]
    assert late[f"alpha{early['component']}"] < -0.2
