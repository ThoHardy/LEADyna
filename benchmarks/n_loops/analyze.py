import glob
import json
import sys

import numpy as np

rows = [r for f in glob.glob(sys.argv[1] if len(sys.argv) > 1 else "rows_*.json")
        for r in json.load(open(f))]
seeds = sorted({r["seed"] for r in rows})
print(f"seeds: {seeds}")
for truth in ["GM", "FG"]:
    for fit in ["GM", "FG"]:
        sub = [r for r in rows if r["truth"] == truth and r["fit"] == fit]
        if not sub:
            continue
        print(f"\n=== data from {truth}, fitted {fit}  (LL per sample, gap to generating model x1e3)")
        loops = [k for k in range(1, 9)] + ["polish"]
        for k in loops:
            rs = [r for r in sub if r["loop"] == k]
            if not rs:
                continue
            tr = np.array([(r["train"] - r["truth_train"]) * 1e3 for r in rs])
            te = np.array([(r["test"] - r["truth_test"]) * 1e3 for r in rs])
            line = f"loop {k!s:>6}: train {tr.mean():+7.2f} ± {tr.std():5.2f}   test {te.mean():+7.2f} ± {te.std():5.2f}"
            if rs[0]["err"]:
                errs = {p: np.mean([abs(r["err"][p]) for r in rs]) for p in rs[0]["err"]}
                keep = ["tau", "threshold"] + (["gain"] if "gain" in errs else []) + \
                    [p for p in errs if p[0] in "wg" and p not in ("w0", "g0", "gain")]
                line += "   |err| " + " ".join(f"{p}={errs[p]:.3f}" for p in keep)
            print(line)
        secs = [r.get("seconds") for r in sub if r["loop"] == "polish"]
        print(f"time for 8 loops: {np.mean(secs):.0f} s")
