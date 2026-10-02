"""Exp 6: layers compared at EQUAL WALL-CLOCK on the static coverage problem.

Same task as Exp 2 (maximise the coverage f through the layer by Adam on a free
score vector, key initialisation), but every method gets the same time budget
for its optimisation steps. Evaluation of the deployed set (one hard greedy call)
is excluded from the budget and done after every step, so the curve is anytime:
best feasible f found within t seconds, for t in --budgets.

Methods: STSG, STSG with soft-sorted costs, perturbed optimiser with M samples
(M in --Ms), randomised greedy (score-function). All share one implementation of
the hard greedy, single-threaded, so wall-clock is comparable within this code.

Comparison rule, fixed before running: at each budget, STSG (or its variant)
"matches" the perturbed optimiser if the paired 95% CI of
gap(perturbed) - gap(STSG) over instances contains 0 or is positive.

  python experiments/exp6_equal_compute.py --out results
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from env.instance import generate
from metrics.stats import paired_compare
from stsg.baselines import make_layer
from stsg.coverage import coverage_torch, coverage_np, milp_multistage, feasible_np


def run(name, kw, inst, budgets, lr, seed):
    torch.manual_seed(seed)
    A = torch.as_tensor(inst.A, dtype=torch.float32)
    w = torch.as_tensor(inst.w / inst.w.sum(), dtype=torch.float32)
    cb = inst.constraint_batch(np.zeros(inst.N), inst.T - 1)
    layer = make_layer(name, **kw)
    key0 = (inst.w[:, None] * inst.A).sum(0) / inst.w.sum()
    y = torch.nn.Parameter(5.0 * torch.as_tensor(key0, dtype=torch.float32).unsqueeze(0)
                           + 0.01 * torch.randn(1, inst.N))
    opt = torch.optim.Adam([y], lr=lr)
    B = float(cb.budget); ones = np.ones(inst.N)
    best, spent, steps = 0.0, 0.0, 0
    curve, bi = {}, 0
    budgets = sorted(budgets)

    def evaluate():
        nonlocal best
        with torch.no_grad():
            dep = layer(y, cb, explore=False).deploy[0].numpy()
        if feasible_np(dep, inst.cost, B, inst.mod_site, inst.cap, ones):
            best = max(best, coverage_np(dep, inst.A, inst.w))

    evaluate()
    while bi < len(budgets):
        t0 = time.perf_counter()
        stoch = getattr(layer, "stochastic", False)
        out = layer(y, cb, explore=stoch)
        if name == "rand_greedy":
            with torch.no_grad():
                f_s = coverage_torch(out.x, A, w)
                f_g = coverage_torch(layer(y, cb, explore=False).deploy, A, w)
            loss = -((f_s - f_g) * out.logp).mean()
        else:
            loss = -coverage_torch(out.x, A, w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        spent += time.perf_counter() - t0
        steps += 1
        evaluate()
        while bi < len(budgets) and spent >= budgets[bi]:
            curve[budgets[bi]] = (best, steps)
            bi += 1
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16])
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    methods = [("stsg", {}), ("stsg_softcost", {})] + [(f"perturbed_M{M}", {"M": M}) for M in a.Ms] + [("rand_greedy", {})]
    recs = []
    for m in a.ms:
        for sd in range(a.seeds):
            inst = generate(m=m, theta=a.theta, seed=10_000 + sd)            # same instances as Exp 2
            opt_val, _, _ = milp_multistage(inst.A, inst.w, inst.cost, [float(inst.stage_budget.sum())],
                                            inst.mod_site, inst.cap)
            for label, kw in methods:
                base = "perturbed" if label.startswith("perturbed") else label
                curve = run(base, kw, inst, a.budgets, a.lr, sd)
                for t, (v, steps) in curve.items():
                    recs.append(dict(m=m, seed=sd, method=label, budget=t, gap=1 - v / opt_val, steps=steps))
            print(f"m={m} seed={sd}: " + " ".join(
                f"{r['method']}={r['gap']:.3f}" for r in recs if r["m"] == m and r["seed"] == sd
                and r["budget"] == max(a.budgets)), flush=True)
            json.dump(recs, open(os.path.join(a.out, "exp6_equal_compute.json"), "w"))
    # summary with the pre-stated comparison rule
    print("\npaired gap(other) - gap(STSG variant); positive = STSG variant better")
    for m in a.ms:
        for t in a.budgets:
            sel = [r for r in recs if r["m"] == m and r["budget"] == t]
            g = lambda meth: {r["seed"]: r["gap"] for r in sel if r["method"] == meth}
            line = [f"m={m} t={t:>4}s"]
            for ref in ("stsg", "stsg_softcost"):
                for other in [f"perturbed_M{M}" for M in a.Ms] + ["rand_greedy"]:
                    c = paired_compare(g(other), g(ref))
                    verdict = "match/better" if c["lo"] <= 0 <= c["hi"] or c["lo"] > 0 else "worse"
                    line.append(f"{other}-{ref}: {c['diff']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] {verdict}")
            print("  " + "\n  ".join(line))
    print("wrote", os.path.join(a.out, "exp6_equal_compute.json"))


if __name__ == "__main__":
    main()
