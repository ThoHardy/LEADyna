import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

rows = sum((json.load(open(f)) for f in sys.argv[1:]), [])
METHODS = ("full", "chunks", "chunks+pre")
LABEL = {"full": "full rest series (no cut, no pre-stim)",
         "chunks": "rest cut in 50-pt chunks (no pre-stim)",
         "chunks+pre": "50-pt chunks + pre-stim (paper)"}
COL = {"full": "#2a6fb0", "chunks": "#9a9a9a", "chunks+pre": "#d1495b"}
BINS = [(2, 4), (4, 7), (7, 13)]


def get(variant, method, key):
    r = [x for x in rows if x["variant"] == variant and x["method"] == method]
    r.sort(key=lambda x: x["seed"])
    return np.array([x["truth"][key] for x in r]), np.array([x["fit"][key] for x in r])


for variant in ("raw", "bc"):
    print(f"\n=== {variant} (n={len(get(variant, 'full', 'tau')[0])}) ===")
    errs = {}
    for m in METHODS:
        t, f = get(variant, m, "tau")
        rel = (f - t) / t
        errs[m] = np.abs(np.log(f / t))
        line = (f"{m:11s} tau: bias {100 * rel.mean():+6.1f}%  MAE {100 * np.abs(rel).mean():5.1f}%  "
                f"RMSE(log) {np.sqrt(np.mean(np.log(f / t) ** 2)):.3f}  rho {spearmanr(t, f)[0]:.3f} | by bin bias: ")
        line += "  ".join(f"[{a},{b}) {100 * rel[(t >= a) & (t < b)].mean():+5.1f}%" for a, b in BINS)
        print(line)
        for k in ("process_noise", "measure_noise"):
            tt, ff = get(variant, m, k)
            print(f"{'':11s} {k}: bias {100 * ((ff - tt) / tt).mean():+5.1f}%  MAE {100 * np.abs((ff - tt) / tt).mean():5.1f}%")
        ws = [get(variant, m, f"w{c}") for c in range(1, 6)]
        tw, fw = np.concatenate([w[0] for w in ws]), np.concatenate([w[1] for w in ws])
        sel = tw > 0.05
        print(f"{'':11s} w (true>0.05): bias {100 * ((fw - tw) / tw)[sel].mean():+5.1f}%  MAE {100 * np.abs((fw - tw) / tw)[sel].mean():5.1f}%")
    for a, b in (("chunks+pre", "full"), ("chunks+pre", "chunks"), ("chunks", "full")):
        d = errs[a] - errs[b]
        print(f"|log err| {a} - {b}: median {np.median(d):+.4f}, {a} better in {np.mean(d < 0) * 100:.0f}%, "
              f"Wilcoxon p={wilcoxon(d)[1]:.2g}")

fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
for i, variant in enumerate(("raw", "bc")):
    for j, m in enumerate(METHODS):
        ax = axes[i, j]
        t, f = get(variant, m, "tau")
        ax.scatter(t, f, s=10, alpha=.6, color=COL[m])
        ax.plot([2, 13], [2, 13], "k--", lw=.8)
        ax.set(xlim=(1.5, 13.5), ylim=(1.5, 15), xlabel="True tau", ylabel="Fitted tau")
        rel = (f - t) / t
        ax.text(.04, .96, f"ρ = {spearmanr(t, f)[0]:.3f}\nbias = {100 * rel.mean():+.1f}%\nMAE = {100 * np.abs(rel).mean():.1f}%",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="w", ec="0.8"))
        ax.set_title(("" if variant == "raw" else "[baseline-corrected] ") + LABEL[m], fontsize=9)
    ax = axes[i, 3]
    edges = np.exp(np.linspace(np.log(2), np.log(13), 7))
    for m in METHODS:
        t, f = get(variant, m, "tau")
        rel = 100 * (f - t) / t
        idx = np.digitize(t, edges) - 1
        cx = [np.sqrt(edges[k] * edges[k + 1]) for k in range(6)]
        mu = [rel[idx == k].mean() for k in range(6)]
        se = [rel[idx == k].std() / np.sqrt((idx == k).sum()) for k in range(6)]
        ax.errorbar(cx, mu, se, color=COL[m], marker="o", ms=4, label=m, capsize=2)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xscale("log")
    ax.set(xlabel="True tau", ylabel="tau relative error (%)", title="Bias vs true tau (mean ± SEM)")
    ax.set_xticks([2, 3, 5, 8, 13], ["2", "3", "5", "8", "13"])
    ax.legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig("tau_recovery_chunks_vs_full.png", dpi=150)
