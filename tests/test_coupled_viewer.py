"""HTML viewer of coupled stages: spec built from a fit, standalone page, input checks."""
import json
import subprocess
import sys

import numpy as np
import pytest

from leadyna import coupled_viewer as cv


def _embedded(page, name):
    start = page.index(f"window.{name} = ") + len(f"window.{name} = ")
    return json.loads(page[start:page.index(";</script>", start)])


def test_viewer_module_does_not_load_jax():
    code = "import sys, leadyna.coupled_viewer; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_demo_page_is_standalone(tmp_path):
    page = cv.export_demo(tmp_path / "demo.html").read_text(encoding="utf-8")
    assert not any(k in page for k in ("__SPEC__", "__BRAIN__", "__TITLE__"))
    spec, brain = _embedded(page, "SPEC"), _embedded(page, "BRAIN")
    assert [c["label"] for c in spec["configs"]] == ["1", "2", "3"]
    assert brain["nLeft"] == 10242
    first = [c["stages"][0] for c in spec["configs"]]          # same stage 1 for every K
    assert all(s["pattern"] == 0 and s["models"] == first[0]["models"] for s in first)
    labels = {p["label"] for p in spec["configs"][2]["presets"]}
    assert {"Stage 2 bistable", "Both bistable"} <= labels


@pytest.fixture(scope="module")
def fitted():
    pytest.importorskip("jax")
    from leadyna import coupled as cl
    stage = dict(tau=5.0, process_noise=0.25, onset=0.04, scale=0.03, amp=0.9)
    sim = cl.simulate_chain([stage], [[1.0]], n_channels=8, n_per_level=40, tmax=0.6, seed=2)
    model = cl.CoupledLEAD(K=1, n_perm=5, n_jobs=1).fit(sim["X"], sim["cats"], sim["times"])
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pos = np.c_[0.08 * np.cos(theta), 0.08 * np.sin(theta), np.full(8, 0.05)]      # metres
    return model, pos


def test_spec_from_fit(fitted, tmp_path):
    model, pos = fitted
    spec = cv.viewer_spec(model, pos, level_labels=["a", "b", "c", "d"])
    stage = spec["configs"][0]["stages"][0]
    fit = model.result_["cascade"]["comps"][0]["fits"]
    assert (spec["dt_ms"], spec["t0"], spec["n_times"]) == (10.0, 50, 110)
    assert set(stage["models"]) == {"linear", "modulated_gain"}
    assert stage["bic_chosen"] == model.summary()[0]["model"]
    assert stage["models"]["linear"]["tau"] == pytest.approx(fit["linear"]["params"]["tau"])
    assert not any(k.startswith("alpha") for m in stage["models"].values() for k in m)
    assert spec["surface"]["kind"] == "scalp"
    assert np.allclose(spec["surface"]["pos"], pos * 1000, atol=0.01)
    assert spec["stimulus"]["levels"] == ["a", "b", "c", "d"]
    page = model.export_html(tmp_path / "fit.html", pos).read_text(encoding="utf-8")
    assert _embedded(page, "SPEC")["configs"][0]["stages"][0]["models"] == stage["models"]


def test_spec_input_checks(fitted):
    model, pos = fitted
    with pytest.raises(ValueError, match="montage"):
        cv.viewer_spec(model, np.zeros_like(pos))
    with pytest.raises(ValueError, match="8"):
        cv.viewer_spec(model, pos[:5])
    with pytest.raises(ValueError, match="level_labels"):
        cv.viewer_spec(model, pos, level_labels=["a"])
    with pytest.raises(ValueError, match="info"):
        cv.viewer_spec(model)
    with pytest.raises(TypeError):
        cv.viewer_spec({"K": 1})
