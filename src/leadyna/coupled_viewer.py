"""Interactive 3-D viewer of a chain of coupled stages, as one standalone HTML file.

The page shows the K patterns on the head, simulates single trials from the fitted stage
parameters (same equations as :mod:`leadyna.coupled`), and lets the viewer change the
stimulus, both noises and the model of each stage (linear or modulated gain), with the
switching threshold of every bistable stage marked on the stimulus scale.

>>> model = CoupledLEAD().fit(X, cats, times)
>>> model.export_html("sub01_coupled.html", info)          # or export_html(model, ...)
>>> export_html({1: m1, 2: m2, 3: m3}, "sub01.html", info)  # one page, K chosen in the page

Patterns are drawn as scalp topographies from the electrode positions in ``info``
(interpolated between electrodes; the template brain inside is for orientation only). If
you have source-space patterns on fsaverage5, pass them as ``cortex_patterns`` instead.
The file loads three.js and two fonts from the web; everything else is inside it.

This module needs only numpy (no JAX): it reads results, it does not fit.
"""
from __future__ import annotations

import json
import webbrowser
from importlib import resources
from pathlib import Path

import numpy as np

__all__ = ["export_html", "viewer_spec", "export_demo", "demo_spec", "render_html"]

PARAM_KEYS = ("tau", "process_noise", "measure_noise", "onset", "scale", "coupling", "delay",
              "g0", "g1", "g2", "threshold")
L, G = "linear", "modulated_gain"


def _result(obj):
    """Accept a fitted :class:`~leadyna.coupled.CoupledLEAD` or the dict of ``fit_coupled``."""
    res = getattr(obj, "result_", obj)
    if not isinstance(res, dict) or "cascade" not in res:
        raise TypeError("Expected a fitted CoupledLEAD or the output of fit_coupled.")
    return res


def _params(p):
    return {k: float(v) for k, v in p.items() if k in PARAM_KEYS or k.startswith("amp")}


def _positions_mm(info, n_channels):
    """Electrode positions in mm (head frame), from an MNE Info, a dict or an array in metres."""
    if hasattr(info, "keys") and "chs" in info:
        names = [ch["ch_name"] for ch in info["chs"]]
        pos = np.array([ch["loc"][:3] for ch in info["chs"]], float)
    elif isinstance(info, dict):
        names, pos = list(info), np.array(list(info.values()), float)
    else:
        pos = np.asarray(info, float)
        names = [f"ch{i + 1}" for i in range(len(pos))]
    if pos.shape != (n_channels, 3):
        raise ValueError(f"Got positions for {len(pos)} channels, but the patterns have "
                         f"{n_channels} (use the info returned by epochs_to_array).")
    if not np.isfinite(pos).all() or (np.linalg.norm(pos, axis=1) < 1e-6).any():
        raise ValueError("Some channels have no position. Set a montage first, e.g. "
                         "epochs.set_montage('standard_1020'), then call epochs_to_array again.")
    return names, (pos * 1000.0).round(2).tolist()


def viewer_spec(results, info=None, *, level_labels=None, stage_names=None, cortex_patterns=None,
                title="CoupledLEAD results", subtitle=None):
    """JSON-ready description of fitted chains for the viewer (see :func:`export_html`)."""
    if isinstance(results, dict) and "cascade" not in results:
        items = sorted(results.items())
    else:
        items = [(_result(results)["K"], results)]
    patterns, configs, timing, amps = [], [], None, []
    for K, obj in items:
        res = _result(obj)
        casc = res["cascade"]
        t = (round(float(casc["dt"]) * 1000, 6), int(casc["t0"]), int(res["Zs"].shape[2]))
        if timing not in (None, t):
            raise ValueError("All results must come from the same epochs (same time axis).")
        timing = t
        if cortex_patterns is not None:
            cols = cortex_patterns[K] if isinstance(cortex_patterns, dict) else cortex_patterns
        else:
            cols = res["A"].T                           # one row per component, discovery order
        stages = []
        for p, i in enumerate(casc["order"]):           # stages in time order
            c = casc["comps"][i]
            patterns.append({"values": np.round(np.asarray(cols[i], float), 6).tolist()})
            models = {m: _params(f["params"]) for m, f in c["fits"].items()}
            if p == 0:
                amps += [v for m in models.values() for k, v in m.items() if k.startswith("amp")]
            name = stage_names[p] if stage_names and p < len(stage_names) else None
            stages.append(dict(name=name, pattern=len(patterns) - 1, models=models,
                               chosen=c["chosen"], bic_chosen=c["chosen"]))
        presets = [dict(label="BIC choice", models=[s["chosen"] for s in stages]),
                   dict(label="All linear", models=[L] * len(stages))]
        if all(len(s["models"]) == 2 for s in stages):
            presets.append(dict(label="All modulated gain", models=[G] * len(stages)))
        configs.append(dict(label=str(K), stages=stages, presets=presets))
    first = next(iter(configs[0]["stages"][0]["models"].values()))
    n_levels = sum(k.startswith("amp") for k in first)
    labels = ([str(v) for v in level_labels] if level_labels is not None
              else [f"level {i + 1}" for i in range(n_levels)])
    if len(labels) != n_levels:
        raise ValueError(f"level_labels has {len(labels)} entries, the fit has {n_levels} "
                         "stimulus levels.")
    if cortex_patterns is not None:
        surface, note = {"kind": "cortex"}, "Patterns on the fsaverage5 cortex, as given."
        footer = "Cortical surface: FreeSurfer fsaverage5. "
    else:
        if info is None:
            raise ValueError("Pass info (electrode positions), e.g. the info returned by "
                             "epochs_to_array.")
        names, pos = _positions_mm(info, len(patterns[0]["values"]))
        surface = {"kind": "scalp", "names": names, "pos": pos}
        note = ("Scalp topographies of the fitted patterns, interpolated between electrodes. "
                "The sign of each pattern goes with the sign of its latent.")
        footer = ("The template brain inside the head is FreeSurfer fsaverage5, scaled to the "
                  "electrodes for orientation only: no source localisation. ")
    n_trials = int(_result(items[0][1])["Zs"].shape[0])
    dt_ms, t0, n_times = timing
    return dict(
        title=title,
        subtitle=subtitle or (f"Chain of coupled 1-D stages fitted on {n_trials} trials. "
                              "Trajectories are simulated from the fitted parameters; latents "
                              "are in rest-SD units."),
        footer=("Made with LEADyna (leadyna.coupled_viewer). " + footer
                + "Stimulus slider: input amplitude of stage 1, fitted levels marked above it. "
                "A switching threshold is the stimulus amplitude above which a bistable stage "
                "ends high without process noise."),
        dt_ms=dt_ms, t0=t0, n_times=n_times, surface=surface, patterns=patterns,
        pattern_note=note, configs=configs, default_config=len(configs) - 1,
        stimulus=dict(label="Stimulus input", max_amp=round(1.25 * max(amps + [1e-6]), 6),
                      levels=labels, default="auto"))


def demo_spec():
    """The illustrative example: 1-3 stages on invented cortical regions (auditory,
    centro-frontal, prefronto-parietal), with parameters chosen for legibility."""
    noise = dict(measure_noise=0.5)
    early = dict(name="early", region="auditory", color="#2ec4b6", pattern=0, models={
        L: dict(tau=5, process_noise=0.25, onset=4, scale=3, **noise)})
    mid_lin = dict(tau=8, process_noise=0.15, coupling=0.15, delay=7, **noise)
    mid = dict(name="middle", region="centro-frontal", color="#a78bfa", pattern=1, models={
        L: mid_lin, G: dict(mid_lin, g0=0.5, g1=0.05, g2=0, threshold=2)})

    def late(c_lin, c_gain, delay):     # a coupling may depend on the model of the parent stage
        lin = dict(tau=10, process_noise=0.12, coupling=c_lin, delay=delay, **noise)
        return dict(name="late", region="prefronto-parietal", color="#ff9f1c", pattern=2,
                    models={L: lin, G: dict(lin, coupling=c_gain, g0=0.4, g1=0.05, g2=0,
                                            threshold=2)})

    late2 = late(0.14, 0.14, 8)
    late3 = late({L: 0.14, G: 0.08}, {L: 0.14, G: 0.027}, 10)
    preset = lambda label, *models: dict(label=label, models=list(models))   # noqa: E731
    configs = [
        dict(label="1", stages=[dict(early, default=L)], presets=[]),
        dict(label="2", stages=[dict(early, default=L), dict(late2, default=G)],
             presets=[preset("Stage 2 bistable", L, G), preset("All linear", L, L)]),
        dict(label="3", stages=[dict(early, default=L), dict(mid, default=L),
                                dict(late3, default=G)],
             presets=[preset("Stage 3 bistable", L, L, G), preset("Stage 2 bistable", L, G, L),
                      preset("Both bistable", L, G, G), preset("All linear", L, L, L)]),
    ]
    regions = [                                           # Gaussian blobs: centre (mm), both sides
        dict(sd=10, at=[[60, -10, 0], [60, -24, 8], [56, -38, 14]]),                  # auditory
        dict(sd=14, at=[[10, -10, 70], [28, -6, 62], [14, 14, 62], [32, 10, 56]]),  # centro-frontal
        dict(sd=13, at=[[42, 38, 22], [30, 52, 12], [46, 22, 36],                     # prefrontal
                        [42, -58, 46], [50, -46, 44], [30, -68, 48]]),               # + parietal
    ]
    return dict(
        title="CoupledLEAD: stages of processing as coupled 1-D latents",
        subtitle=("The recorded signal is a sum of fixed spatial patterns, each scaled by its own "
                  "noisy latent: x(t) = Σ aₖ zₖ(t) + noise. The stimulus drives stage 1; each "
                  "later stage is driven by the previous one after a delay."),
        footer=("Illustrative simulation. Cortical surface: FreeSurfer fsaverage5; cerebellum and "
                "brainstem are schematic. The regions are chosen for display, not estimated from "
                "data. Dynamics: the CoupledLEAD stage equations at 100 Hz, with parameters chosen "
                "for legibility (couplings into stage 3 depend on the model of stage 2). A "
                "switching threshold is the stimulus strength above which a bistable stage ends "
                "high without process noise."),
        dt_ms=10, t0=20, n_times=120, surface={"kind": "cortex"},
        patterns=[{"blobs": r} for r in regions],
        pattern_note=("Each pattern lives in its own regions. Stage 1 keeps the same pattern "
                      "whatever the number of stages."),
        stimulus=dict(label="Stimulus strength", max_amp=0.9, levels=None, default=0.55),
        configs=configs, default_config=1)


def render_html(spec):
    """Fill the viewer template with ``spec`` and the fsaverage5 surface."""
    files = resources.files("leadyna") / "_viewer"
    page = (files / "coupled_viewer.html").read_text(encoding="utf-8")
    brain = (files / "fsaverage5.json").read_text(encoding="utf-8")
    spec_js = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = spec["title"].replace("&", "&amp;").replace("<", "&lt;")
    return (page.replace("__TITLE__", title).replace("__SPEC__", spec_js, 1)
            .replace("__BRAIN__", brain, 1))


def _write(spec, path, open_browser):
    path = Path(path)
    path.write_text(render_html(spec), encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def export_html(results, path, info=None, *, level_labels=None, stage_names=None,
                cortex_patterns=None, title="CoupledLEAD results", subtitle=None,
                open_browser=False):
    """Write the interactive viewer of fitted chain(s) to ``path`` (one standalone .html file).

    ``results``: a fitted :class:`~leadyna.coupled.CoupledLEAD`, the dict of
    :func:`~leadyna.coupled.fit_coupled`, or ``{K: either}`` to choose K inside the page
    (all fitted on the same epochs). ``info``: electrode positions, i.e. the MNE ``info``
    returned by :func:`~leadyna.frontends.eeg_mne.epochs_to_array` (a montage must be set),
    a dict ``{name: (x, y, z)}`` or an array (channels, 3), in metres. ``level_labels``: one
    label per stimulus level (e.g. SNRs in dB). ``stage_names``: optional, in time order.
    ``cortex_patterns``: (components, 20484) values on fsaverage5, instead of scalp patterns.
    Both fitted models of every stage (linear, modulated gain) can be chosen in the page;
    the BIC choice is the default. ``open_browser=True`` opens the file. Returns the path.
    """
    spec = viewer_spec(results, info, level_labels=level_labels, stage_names=stage_names,
                       cortex_patterns=cortex_patterns, title=title, subtitle=subtitle)
    return _write(spec, path, open_browser)


def export_demo(path, open_browser=False):
    """Write the illustrative demo (no data needed). Returns the path."""
    return _write(demo_spec(), path, open_browser)
