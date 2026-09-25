"""Exp 3: main table (Table tab:main) on the MCS-BSS corridor.

1. train the amortised follower once per instance (checkpoint reused by all methods)
2. NashConv (restricted deviations, lower bound) on sampled configurations
3. calibrate reference costs with the re-ranking coverage greedy
4. train the constrained leader with every action layer, same seeds / budget
5. evaluate: violation rate, discounted return, grid / delay / CVaR, coverage gap
   to the multistage MILP (coverage component only: the MILP cannot see the
   follower game, so "gap to MILP" must be labelled as coverage gap in the paper)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from agents.follower import FollowerPolicy, train_follower
from agents.leader import LeaderAgent, LeaderConfig, calibrate
from env.instance import generate, random_feasible_config
from env.leader_env import LeaderEnv
from metrics.nashconv import nashconv
from stsg.coverage import milp_multistage

MAIN = ["penalty", "lagrangian", "qp_round", "gumbel_topk", "rand_greedy", "perturbed", "stsg"]
ABL = ["abl_no_matroid_gate", "abl_naive_softsort", "abl_soft_forward", "stsg_density", "stsg_softcost"]


def get_follower(inst, path, iters, episodes, seed):
    if os.path.exists(path):
        pol = FollowerPolicy(); pol.load_state_dict(torch.load(path)); return pol, None
    pol, log = train_follower(inst, iters=iters, episodes=episodes, seed=seed)
    torch.save(pol.state_dict(), path)
    return pol, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--inst_seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=MAIN)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--no_cvar", action="store_true", help="ablation e")
    ap.add_argument("--follower_iters", type=int, default=150)
    ap.add_argument("--follower_eps", type=int, default=8)
    ap.add_argument("--leader_iters", type=int, default=150)
    ap.add_argument("--leader_eps", type=int, default=16)
    ap.add_argument("--nashconv_configs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1, help="parallel (method, seed) jobs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"m{a.m}_th{a.theta:g}_T{a.T}_i{a.inst_seed}"
    inst = generate(m=a.m, theta=a.theta, T=a.T, seed=a.inst_seed)

    t0 = time.time()
    follower, flog = get_follower(inst, os.path.join(a.out, f"follower_{tag}.pt"),
                                  a.follower_iters, a.follower_eps, a.inst_seed)
    f_time = time.time() - t0

    rng = np.random.default_rng(123)
    nc = [nashconv(inst, follower, random_feasible_config(inst, int(rng.integers(inst.T)), rng), seed=i)
          for i in range(a.nashconv_configs)]
    print(f"[nashconv] mean {np.mean([n['nashconv'] for n in nc]):.3f}  rel {np.mean([n['rel'] for n in nc]):.3f} (lower bound)")

    env0 = LeaderEnv(inst, follower, seed=0)
    ref = calibrate(env0)
    milp_val, milp_X, msg = milp_multistage(inst.A, inst.w, inst.cost, inst.stage_budget,
                                            inst.mod_site, inst.cap, gamma=0.95)
    print(f"[ref] {ref}  MILP discounted coverage {milp_val:.3f} ({msg})")

    methods = list(a.methods) + (ABL if a.ablations else [])
    results = {"tag": tag, "ref": ref, "milp_cov_disc": milp_val, "follower_time": f_time,
               "follower_log": flog, "nashconv": nc, "runs": []}
    jobs = [(meth, sd) for meth in methods for sd in a.seeds]
    fpath = os.path.join(a.out, f"follower_{tag}.pt")
    args = [(inst, fpath, meth, sd, a.leader_iters, a.leader_eps, not a.no_cvar, ref, milp_val, a.workers == 1)
            for meth, sd in jobs]
    out_path = os.path.join(a.out, f"exp3_{tag}{'_nocvar' if a.no_cvar else ''}.json")
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
            for run in ex.map(run_one, args):
                results["runs"].append(run); report(run)
                json.dump(results, open(out_path, "w"))
    else:
        for arg in args:
            run = run_one(arg); results["runs"].append(run); report(run)
            json.dump(results, open(out_path, "w"))
    print("wrote", out_path)


def report(run):
    ev = run["eval"]
    print(f"== {run['method']} seed {run['seed']}: ret {ev['ret']:.3f} infeas {ev['infeas_rate']:.2f} "
          f"covgap {ev['cov_gap_milp']:.3f} grid {ev['grid']:.2f} cvar {ev['cvar_grid']:.2f}", flush=True)


def run_one(arg):
    inst, fpath, meth, sd, iters, eps, use_cvar, ref, milp_val, verbose = arg
    torch.set_num_threads(1)
    follower = FollowerPolicy(); follower.load_state_dict(torch.load(fpath))
    env = LeaderEnv(inst, follower, seed=sd)
    cfg = LeaderConfig(layer=meth, iters=iters, episodes_per_iter=eps, seed=sd, use_cvar=use_cvar)
    agent = LeaderAgent(env, cfg, ref)
    t1 = time.time()
    log = agent.train(verbose=verbose, log_every=max(1, iters // 5))
    ev = agent.evaluate()
    ev["cov_gap_milp"] = 1.0 - ev["cov_disc"] / milp_val if milp_val > 0 else float("nan")
    return {"method": meth, "seed": sd, "train_time": time.time() - t1,
            "time_per_update": (time.time() - t1) / iters, "eval": ev, "log": log, "mu": agent.mu}


if __name__ == "__main__":
    main()
