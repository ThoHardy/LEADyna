import json
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

rows = sum((json.load(open(f)) for f in sys.argv[1:]), [])
METHODS = ("full", "chunks", "chunks+pre")


def get(variant, method, init, key="tau"):
    r = sorted((x for x in rows if (x["variant"], x["method"], x["init"]) == (variant, method, init)),
               key=lambda x: x["seed"])
    return np.array([x["truth"][key] for x in r]), np.array([x["fit"][key] for x in r])


for variant in ("raw", "bc"):
    for init in ("leadyna", "stationary"):
        print(f"\n=== {variant} / init={init} (n={len(get(variant, 'full', init)[0])}) ===")
        errs = {}
        for m in METHODS:
            t, f = get(variant, m, init)
            rel = (f - t) / t
            errs[m] = np.abs(np.log(f / t))
            sd = np.std(np.log(f / t))
            print(f"{m:11s} tau bias {100 * rel.mean():+6.1f}%  MAE {100 * np.abs(rel).mean():5.1f}%  "
                  f"SD(log err) {sd:.3f}  rho {spearmanr(t, f)[0]:.4f}  | "
                  + "  ".join(f"{k}: {100 * ((get(variant, m, init, k)[1] - get(variant, m, init, k)[0]) / get(variant, m, init, k)[0]).mean():+5.1f}%"
                              for k in ("process_noise", "measure_noise")))
        d = errs["chunks+pre"] - errs["full"]
        print(f"chunks+pre vs full: better in {np.mean(d < 0) * 100:.0f}%, Wilcoxon p={wilcoxon(d)[1]:.2g}")
