"""Exp 8: plan-search reference with simulator access (item 4).

Cross-entropy search over open-loop plans. A plan is one score vector per stage,
y_t in R^N, decoded stage by stage with the hard greedy (Algorithm 1) on the
residual budget and capacities, so every candidate is feasible for X_t by
construction. Each candidate is scored with evaluate_policy on the same K demand
quantiles, frozen follower and CMDP definition as the learners, i.e. the search
optimises the evaluation objective directly. It is therefore an ORACLE REFERENCE
(an estimate of what the plan space admits), not a competing learner.

fitness = return                                  if CMDP-feasible
        = return - penalty * (max_h g_h/beta_h - 1)   otherwise
Reported: best CMDP-feasible plan found, and the best plan overall.

  python experiments/exp8_plan_search.py --m 16 --inst_seed 0 --out results_m16
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from agents.follower import FollowerPolicy
from agents.leader import LeaderConfig, calibrate, evaluate_policy
from env.instance import generate
from env.leader_env import LeaderEnv
from stsg.coverage import static_key_greedy


def plan_policy(Y):
    """Open-loop plan: stage t decodes scores Y[t] with the hard greedy."""
    def act(state, env):
        inst = env.inst
        budget, cap, avail = inst.residual(env.built, env.t)
        return static_key_greedy(Y[env.t], inst.cost, budget, inst.mod_site, cap, avail)
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--inst_seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--penalty", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta_grid", type=float, default=LeaderConfig.beta_grid)
    ap.add_argument("--beta_delay", type=float, default=LeaderConfig.beta_delay)
    ap.add_argument("--beta_cvar", type=float, default=LeaderConfig.beta_cvar)
    ap.add_argument("--out", default="results_m16")
    a = ap.parse_args()
    torch.set_num_threads(1)
    tag = f"m{a.m}_th{a.theta:g}_T{a.T}_i{a.inst_seed}"
    inst = generate(m=a.m, theta=a.theta, T=a.T, seed=a.inst_seed)
    fpath = os.path.join(a.out, f"follower_{tag}.pt")
    if not os.path.exists(fpath):
        raise SystemExit(f"follower checkpoint {fpath} not found: run exp3 (or its --probe) first")
    fol = FollowerPolicy(); fol.load_state_dict(torch.load(fpath))
    env = LeaderEnv(inst, fol, seed=0)
    ref = calibrate(env)
    old = os.path.join(a.out, f"exp3_{tag}.json")
    if os.path.exists(old):                       # make sure the normalisation matches exp3
        stored = json.load(open(old))["ref"]
        if any(abs(stored[k] - ref[k]) > 1e-6 * max(1, abs(ref[k])) for k in ref):
            raise SystemExit("cost references differ from the stored exp3 run; wrong follower checkpoint?")
    cfg = LeaderConfig(beta_grid=a.beta_grid, beta_delay=a.beta_delay, beta_cvar=a.beta_cvar)

    rng = np.random.default_rng(a.seed)
    key = (inst.w[:, None] * inst.A).sum(0)
    key = key / key.max()
    mu = np.tile(key, (a.T, 1))                    # start at the static coverage key
    sd = np.full((a.T, inst.N), 1.0)
    best_ok, best_any, hist, n_eval = None, None, [], 0
    t0 = time.time()
    for it in range(a.iters):
        pop = [mu + sd * rng.standard_normal(mu.shape) for _ in range(a.pop)]
        if it == 0:
            pop[0] = mu.copy()
        scored = []
        for Y in pop:
            ev = evaluate_policy(env, plan_policy(Y), cfg.gamma, cfg.alpha_cvar, ref=ref, cfg=cfg)
            n_eval += 1
            fit = ev["ret"] - (0 if ev["cmdp_ok"] else a.penalty * max(0.0, ev["cmdp_max_ratio"] - 1))
            scored.append((fit, Y, ev))
            if ev["cmdp_ok"] and (best_ok is None or ev["ret"] > best_ok["ret"]):
                best_ok = dict(ev, iteration=it)
            if best_any is None or ev["ret"] > best_any["ret"]:
                best_any = dict(ev, iteration=it)
        scored.sort(key=lambda z: -z[0])
        E = np.stack([Y for _, Y, _ in scored[: a.elite]])
        mu, sd = E.mean(0), np.maximum(E.std(0), 0.05)
        hist.append(dict(it=it, best_fit=scored[0][0], best_ok_ret=None if best_ok is None else best_ok["ret"],
                         evals=n_eval, time=time.time() - t0))
        bo = "none" if best_ok is None else "%.3f" % best_ok["ret"]
        print(f"[cem] it {it:2d} best fitness {scored[0][0]:.3f}  best CMDP-feasible return {bo}  "
              f"({n_eval} evals, {time.time() - t0:.0f}s)", flush=True)
    keep = ("ret", "g_grid", "g_delay", "g_cvar", "cmdp_ok", "cmdp_max_ratio", "cov_disc", "built_example",
            "iteration", "served_frac")
    out = {"tag": tag, "ref": ref, "betas": dict(beta_grid=a.beta_grid, beta_delay=a.beta_delay,
                                                  beta_cvar=a.beta_cvar),
           "best_feasible": None if best_ok is None else {k: best_ok[k] for k in keep},
           "best_any": {k: best_any[k] for k in keep}, "history": hist, "evals": n_eval,
           "config": vars(a)}
    path = os.path.join(a.out, f"exp8_plan_search_{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    bf = out["best_feasible"]
    msg = "none found" if bf is None else "return %.3f, built %d" % (bf["ret"], int(sum(bf["built_example"])))
    print("best CMDP-feasible plan:", msg)
    print("wrote", path)


if __name__ == "__main__":
    main()
