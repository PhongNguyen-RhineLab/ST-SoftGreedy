"""One line per (config dir, method): per-seed return, max g/beta, built modules.

  python experiments/pilot_summary.py results_m16 results_m16_ws results_m16_ex results_m16_wsex
"""
import glob
import json
import os
import sys

import numpy as np

for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "exp3_*.json"))):
        r = json.load(open(f))
        print(f"== {f}  train_kw={r.get('train_kw')}  betas={r.get('betas')}")
        for meth in dict.fromkeys(x["method"] for x in r["runs"]):
            rr = sorted((x for x in r["runs"] if x["method"] == meth), key=lambda x: x["seed"])
            ev = [x["eval"] for x in rr]
            ok = sum(int(e.get("cmdp_ok", 0)) for e in ev)
            seeds = " ".join(f"s{x['seed']}:{e['ret']:.2f}/{e.get('cmdp_max_ratio', float('nan')):.2f}/{int(sum(e['built_example']))}"
                             for x, e in zip(rr, ev))
            print(f"   {meth:22s} n={len(rr):2d} ret {np.mean([e['ret'] for e in ev]):.3f} "
                  f"cmdp_ok {ok}/{len(rr)}   [ret/max g/beta/built] {seeds}")
