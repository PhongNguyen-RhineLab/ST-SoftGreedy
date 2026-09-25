"""Exp 4: 'hypergradient bias under amortisation versus per-configuration retraining'.

The leader's decision map is piecewise constant, so a hypergradient in the
classical sense does not exist here (see Remark after Theorem consistency).
What the leader actually consumes are follower outcomes g(X) and their
discrete marginals  Delta_j g(X) = g(X ∪ {j}) - g(X).  We measure the bias of
both under the amortised follower against a follower retrained on each fixed X:
  MAE of g, MAE of Delta_j, sign agreement of Delta_j, Kendall tau of the
  ranking of candidate modules by Delta_j.
Suggested wording for the paper: 'amortisation bias in outcomes and in
marginal module values', not 'hypergradient bias'.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import kendalltau

from agents.follower import FollowerPolicy, train_follower, run_episode
from env.instance import generate, random_feasible_config
from env.operations import CorridorOps
from stsg.coverage import feasible_np

KEYS = ("served_frac", "grid", "delay", "profit")


def outcome(inst, pol, x, scales):
    env = CorridorOps(inst, np.random.default_rng(0))
    mets = [run_episode(env, pol, x, True, s)[1] for s in scales]
    return {k: float(np.mean([m[k] for m in mets])) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--base_configs", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--amort_iters", type=int, default=150)
    ap.add_argument("--retrain_iters", type=int, default=60)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = generate(m=a.m, theta=a.theta, T=a.T, seed=0)
    tag = f"m{a.m}_th{a.theta:g}_T{a.T}_i0"
    fpath = os.path.join(a.out, f"follower_{tag}.pt")
    if os.path.exists(fpath):
        amort = FollowerPolicy(); amort.load_state_dict(torch.load(fpath))
    else:
        amort, _ = train_follower(inst, iters=a.amort_iters, seed=0)
        torch.save(amort.state_dict(), fpath)
    scales = np.exp(0.25 * np.array([-1.15, -0.32, 0.32, 1.15]))
    rng = np.random.default_rng(7)
    B = inst.stage_budget.sum(); ones = np.ones(inst.N)
    recs = []
    for k in range(a.base_configs):
        # leave room so that single-module additions stay feasible
        X = random_feasible_config(inst, max(0, inst.T - 2), rng)
        cands = [j for j in rng.permutation(inst.N) if X[j] == 0 and
                 feasible_np(np.where(np.arange(inst.N) == j, 1.0, X), inst.cost, B, inst.mod_site, inst.cap, ones)]
        cands = cands[: a.candidates]
        configs = [("base", X)] + [(int(j), np.where(np.arange(inst.N) == j, 1.0, X)) for j in cands]
        for name, x in configs:
            pol_r, _ = train_follower(inst, iters=a.retrain_iters, fixed_config=x, seed=100 + k, verbose=False)
            recs.append({"base": k, "cand": name, "amort": outcome(inst, amort, x, scales),
                         "retrain": outcome(inst, pol_r, x, scales)})
            print(f"base {k} cand {name}: amort {recs[-1]['amort']['served_frac']:.3f} "
                  f"retrain {recs[-1]['retrain']['served_frac']:.3f}", flush=True)
    summ = {}
    for key in KEYS:
        err = [abs(r["amort"][key] - r["retrain"][key]) for r in recs]
        da, dr = [], []
        taus = []
        for k in range(a.base_configs):
            rs = [r for r in recs if r["base"] == k]
            b = rs[0]
            ga = [r["amort"][key] - b["amort"][key] for r in rs[1:]]
            gr = [r["retrain"][key] - b["retrain"][key] for r in rs[1:]]
            da += ga; dr += gr
            if len(ga) > 1:
                taus.append(kendalltau(ga, gr).statistic)
        da, dr = np.array(da), np.array(dr)
        summ[key] = {"mae_outcome": float(np.mean(err)),
                     "mae_marginal": float(np.mean(np.abs(da - dr))),
                     "sign_agree": float(np.mean(np.sign(da) == np.sign(dr))),
                     "kendall_tau": float(np.nanmean(taus)) if taus else float("nan")}
    print(json.dumps(summ, indent=1))
    json.dump({"records": recs, "summary": summ}, open(os.path.join(a.out, f"exp4_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
