"""Exp 2 (isolates the action layer, no follower, no critic).

For each instance: free score vector y in R^N, maximise the monotone submodular
coverage f through the layer by Adam (pathwise for ST layers, REINFORCE for
rand_greedy, penalty kappa * violation for Bernoulli). Report deployment
violation rate, coverage gap to the exact MILP optimum, and wall-clock.
Classical references: re-ranking greedy (gain / density / best) and Algorithm 1
with the static modular key sum_d w_d A_dj, which quantifies the static-vs-
dynamic ranking gap flagged for Theorem transfer.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from env.instance import generate
from stsg.baselines import make_layer
from stsg.coverage import (coverage_torch, coverage_np, rerank_greedy, static_key_greedy,
                           milp_multistage, feasible_np)

METHODS = ["stsg", "stsg_density", "stsg_softcost", "penalty", "lagrangian", "qp_round", "gumbel_topk",
           "rand_greedy", "perturbed", "abl_no_matroid_gate", "abl_naive_softsort",
           "abl_soft_forward", "abl_paper_gate_offset"]


def optimise(name, inst, steps, lr, seed, kappa=5.0, init="key"):
    """Returns (final deployed x, best feasible deployed x over iterates, wall).
    'best' is the anytime result: every deployed iterate is checked exactly."""
    torch.manual_seed(seed)
    A = torch.as_tensor(inst.A, dtype=torch.float32)
    w = torch.as_tensor(inst.w / inst.w.sum(), dtype=torch.float32)   # f normalised to [0,1]
    cb = inst.constraint_batch(np.zeros(inst.N), inst.T - 1)   # full horizon budget, single stage
    layer = make_layer(name)
    key0 = (inst.w[:, None] * inst.A).sum(0) / inst.w.sum()
    y0 = 5.0 * torch.as_tensor(key0, dtype=torch.float32).unsqueeze(0) if init == "key" else torch.zeros(1, inst.N)
    y = torch.nn.Parameter(y0 + 0.01 * torch.randn(1, inst.N))
    ones = np.ones(inst.N); B = float(cb.budget)
    best_x, best_v = np.zeros(inst.N), -1.0
    opt = torch.optim.Adam([y], lr=lr)
    mu = kappa if name == "penalty" else 0.0
    best = None
    t0 = time.time()
    for it in range(steps):
        stoch = getattr(layer, "stochastic", False)
        out = layer(y, cb, explore=stoch)
        viol = cb.violation(out.x)["total"] / cb.budget.clamp_min(1e-6)   # relative excess
        if name == "rand_greedy":
            with torch.no_grad():
                f_s = coverage_torch(out.x, A, w)
                f_g = coverage_torch(layer(y, cb, explore=False).deploy, A, w)
            loss = -((f_s - f_g) * out.logp).mean()
        else:
            loss = -(coverage_torch(out.x, A, w) - mu * viol).mean()
        with torch.no_grad():
            dep = layer(y, cb, explore=False).deploy[0].numpy()
        if feasible_np(dep, inst.cost, B, inst.mod_site, inst.cap, ones):
            v = coverage_np(dep, inst.A, inst.w)
            if v > best_v:
                best_v, best_x = v, dep
        opt.zero_grad(); loss.backward(); opt.step()
        if name == "lagrangian":
            mu = max(0.0, mu + 0.05 * float(viol.detach().mean()))
    wall = time.time() - t0
    with torch.no_grad():
        dep = layer(y, cb, explore=False).deploy[0].numpy()
    return dep, best_x, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--thetas", type=float, nargs="+", default=[2, 5, 10])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--init", default="key", choices=["key", "zero"])
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.5, 2.0, 5.0],
                    help="penalty weights tried for the penalty baseline (best kept)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = []
    for m in a.ms:
        for th in a.thetas:
            for sd in range(a.seeds):
                inst = generate(m=m, theta=th, seed=10_000 + sd)
                B = float(inst.stage_budget.sum()); ones = np.ones(inst.N)
                t0 = time.time()
                opt_val, X, msg = milp_multistage(inst.A, inst.w, inst.cost, [B], inst.mod_site, inst.cap)
                milp_t = time.time() - t0
                base = dict(m=m, theta=th, seed=sd, N=inst.N, opt=opt_val, milp_time=milp_t)
                refs = {
                    "greedy_gain": rerank_greedy(inst.A, inst.w, inst.cost, B, inst.mod_site, inst.cap, ones, rule="gain"),
                    "greedy_density": rerank_greedy(inst.A, inst.w, inst.cost, B, inst.mod_site, inst.cap, ones, rule="density"),
                    "greedy_best": rerank_greedy(inst.A, inst.w, inst.cost, B, inst.mod_site, inst.cap, ones, rule="best"),
                    "static_key": static_key_greedy((inst.w[:, None] * inst.A).sum(0), inst.cost, B, inst.mod_site, inst.cap, ones),
                }
                for k, x in refs.items():
                    recs.append({**base, "method": k, "val": coverage_np(x, inst.A, inst.w),
                                 "feasible": feasible_np(x, inst.cost, B, inst.mod_site, inst.cap, ones), "wall": 0.0})
                for meth in a.methods:
                    cands = []
                    for kap in (a.kappas if meth == "penalty" else [5.0]):
                        x, xb, wall = optimise(meth, inst, a.steps, a.lr, sd, kappa=kap, init=a.init)
                        feas = feasible_np(x, inst.cost, B, inst.mod_site, inst.cap, ones)
                        cands.append((feas, coverage_np(x, inst.A, inst.w), x, xb, wall, kap))
                    feas, val, x, xb, wall, kap = max(cands, key=lambda c: (c[0], c[1]))
                    recs.append({**base, "method": meth, "val": val, "feasible": feas, "wall": wall,
                                 "val_best_iterate": coverage_np(xb, inst.A, inst.w), "kappa": kap})
                print(f"m={m} theta={th} seed={sd} opt={opt_val:.3f} " +
                      " ".join(f"{r['method']}={r['val']:.2f}{'' if r['feasible'] else '*'}"
                               for r in recs if r['m'] == m and r['theta'] == th and r['seed'] == sd))
    for r in recs:
        r["gap"] = 1.0 - r["val"] / r["opt"] if r["opt"] > 0 else 0.0
    path = os.path.join(a.out, "exp2_static.json")
    json.dump(recs, open(path, "w"), indent=1)
    print("wrote", path, "(* = infeasible deployment; its gap is not comparable)")


if __name__ == "__main__":
    main()
