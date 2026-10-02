"""Exp 3: main table (Table tab:main) on the MCS-BSS corridor.

1. train the amortised follower once per instance (checkpoint reused by all methods)
2. NashConv (restricted deviations, lower bound) on sampled configurations
3. calibrate reference costs with the re-ranking coverage greedy
4. train the constrained leader with every action layer, same seeds / budget
5. evaluate: violation rate, discounted return, grid / delay / CVaR, coverage gap
   to the multistage MILP (coverage component only: the MILP cannot see the
   follower game, so "gap to MILP" must be labelled as coverage gap in the paper)
6. evaluate the non-learned reference plans with the same protocol
   (agents.leader.reference_policies), stored under results["references"]

Results are MERGED into results/exp3_<tag>[_nocvar].json: runs with the same
(method, seed) are skipped (or replaced with --rerun), other runs are kept.
The merge is refused if the cost references differ from the stored ones,
because returns under different normalisations are not comparable.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from agents.follower import FollowerPolicy, train_follower
from agents.leader import (LeaderAgent, LeaderConfig, calibrate, evaluate_policy, reference_policies,
                           constrained_greedy_reference)
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
    ap.add_argument("--rerun", action="store_true", help="replace stored (method, seed) runs instead of skipping")
    ap.add_argument("--beta_grid", type=float, default=LeaderConfig.beta_grid)
    ap.add_argument("--beta_delay", type=float, default=LeaderConfig.beta_delay)
    ap.add_argument("--beta_cvar", type=float, default=LeaderConfig.beta_cvar)
    ap.add_argument("--delay_metric", default="per_served", choices=["per_served", "total"])
    ap.add_argument("--explore_sigma", type=float, default=LeaderConfig.explore_sigma)
    ap.add_argument("--explore_decay", type=float, default=LeaderConfig.explore_decay)
    ap.add_argument("--lr_dual", type=float, default=LeaderConfig.lr_dual)
    ap.add_argument("--layer_kw", default="{}",
                    help='JSON kwargs for the action layer, e.g. \'{"M": 4}\' for the perturbed optimiser')
    ap.add_argument("--prior_scale", type=float, default=LeaderConfig.prior_scale,
                    help="warm start: actor = residual on the static coverage key (0 = off)")
    ap.add_argument("--probe", action="store_true",
                    help="train/load follower, calibrate, print the heuristic constraint landscape, exit")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"m{a.m}_th{a.theta:g}_T{a.T}_i{a.inst_seed}"
    inst = generate(m=a.m, theta=a.theta, T=a.T, seed=a.inst_seed)

    t0 = time.time()
    follower, flog = get_follower(inst, os.path.join(a.out, f"follower_{tag}.pt"),
                                  a.follower_iters, a.follower_eps, a.inst_seed)
    f_time = time.time() - t0

    out_path = os.path.join(a.out, f"exp3_{tag}{'_nocvar' if a.no_cvar else ''}.json")
    old = json.load(open(out_path)) if os.path.exists(out_path) else None

    betas = dict(beta_grid=a.beta_grid, beta_delay=a.beta_delay, beta_cvar=a.beta_cvar)
    train_kw = dict(explore_sigma=a.explore_sigma, explore_decay=a.explore_decay, prior_scale=a.prior_scale)
    if a.lr_dual != LeaderConfig.lr_dual:          # only recorded when changed, so older result files still merge
        train_kw["lr_dual"] = a.lr_dual
    layer_kw = json.loads(a.layer_kw)
    if layer_kw:
        train_kw["layer_kw"] = layer_kw
    env0 = LeaderEnv(inst, follower, seed=0, delay_metric=a.delay_metric)
    ref = calibrate(env0)
    if old is not None and not a.probe:
        old_train = old.get("train_kw", dict(explore_sigma=0.3, explore_decay=0.99, prior_scale=0.0))
        if old_train != train_kw:
            raise SystemExit(f"stored runs used {old_train}; got {train_kw}. Use a new --out.")
        if old.get("betas") not in (None, betas) or old.get("delay_metric", a.delay_metric) != a.delay_metric:
            raise SystemExit(f"stored runs used betas={old.get('betas')} delay_metric={old.get('delay_metric')}; "
                             f"got {betas} {a.delay_metric}. Use a new --out.")
        same_keys = set(old["ref"]) == set(ref)
        drift = max(abs(old["ref"][k] - ref[k]) / max(abs(ref[k]), 1e-9) for k in ref) if same_keys else 1.0
        if drift > 1e-6:
            raise SystemExit(f"cost references changed ({old['ref']} -> {ref}); the follower checkpoint "
                             f"or the cost definition differs from the one used for stored runs. Use a new --out or delete {out_path}.")
    milp_val, milp_X, msg = milp_multistage(inst.A, inst.w, inst.cost, inst.stage_budget,
                                            inst.mod_site, inst.cap, gamma=0.95)
    print(f"[ref] {ref}  MILP discounted coverage {milp_val:.3f} ({msg})")

    if a.probe:
        nc = []
    elif old is not None and old.get("nashconv"):
        nc = old["nashconv"]
    else:
        rng = np.random.default_rng(123)
        nc = [nashconv(inst, follower, random_feasible_config(inst, int(rng.integers(inst.T)), rng), seed=i)
              for i in range(a.nashconv_configs)]
    if nc:
        print(f"[nashconv] mean {np.mean([n['nashconv'] for n in nc]):.3f}  "
              f"rel {np.mean([n['rel'] for n in nc]):.3f} (lower bound)")

    rcfg = LeaderConfig(use_cvar=not a.no_cvar, **betas)   # same betas / gamma / alpha as the learners
    refs = {}
    for name, fn in reference_policies().items():
        refs[name] = evaluate_policy(env0, fn, rcfg.gamma, rcfg.alpha_cvar, ref=ref, cfg=rcfg)
    refs["ref_greedy_constrained"], sweep = constrained_greedy_reference(env0, ref, rcfg)
    for name, ev in refs.items():
        ev["cov_gap_milp"] = 1.0 - ev["cov_disc"] / milp_val if milp_val > 0 else float("nan")
        print(f"== {name}{' ' + ev['family'] + ' phi=' + str(ev['phi']) if 'phi' in ev else ''}: ret {ev['ret']:.3f} "
              f"covgap {ev['cov_gap_milp']:.3f} g/beta grid {ev['g_grid']:.2f} delay {ev['g_delay']:.2f} "
              f"cvar {ev['g_cvar']:.2f} cmdp_ok {int(ev['cmdp_ok'])}")
    for fam in dict.fromkeys(e["family"] for e in sweep):
        print(f"[phi sweep {fam}] " + " ".join(f"{e['phi']}:{e['ret']:.2f}{'' if e['cmdp_ok'] else '*'}"
                                             for e in sweep if e["family"] == fam) + "   (* = violates CMDP)")
    if a.probe:
        print("[probe] g/beta per plan (grid / delay / cvar):")
        for e in sweep:
            print(f"   {e['family']:12s} phi={e['phi']:.1f} built {e['n_built']:3d} ret {e['ret']:.3f}  "
                  f"{e['g_grid']:.2f} / {e['g_delay']:.2f} / {e['g_cvar']:.2f}  {'OK' if e['cmdp_ok'] else ''}")
        cov = refs["ref_greedy_rerank"]
        print(f"[probe] coverage greedy binds: {not cov['cmdp_ok']}   "
              f"some NON-EMPTY heuristic feasible: {any(e['cmdp_ok'] and e['n_built'] > 0 for e in sweep)}")
        return

    results = {"tag": tag, "ref": ref, "milp_cov_disc": milp_val, "follower_time": f_time,
               "follower_log": flog if flog is not None else (old or {}).get("follower_log"),
               "nashconv": nc, "references": refs, "phi_sweep": sweep, "betas": betas, "train_kw": train_kw,
               "delay_metric": a.delay_metric, "runs": list((old or {}).get("runs", []))}
    methods = list(dict.fromkeys(list(a.methods) + (ABL if a.ablations else [])))
    done = {(r["method"], r["seed"]) for r in results["runs"]}
    jobs = [(meth, sd) for meth in methods for sd in a.seeds]
    if a.rerun:
        results["runs"] = [r for r in results["runs"] if (r["method"], r["seed"]) not in set(jobs)]
    else:
        skipped = [j for j in jobs if j in done]
        jobs = [j for j in jobs if j not in done]
        if skipped:
            print(f"[merge] skipping {len(skipped)} stored runs, {len(jobs)} to run")
    json.dump(results, open(out_path, "w"))
    fpath = os.path.join(a.out, f"follower_{tag}.pt")
    adir = os.path.join(a.out, "actors")
    os.makedirs(adir, exist_ok=True)
    args = [(inst, fpath, meth, sd, a.leader_iters, a.leader_eps, not a.no_cvar, ref, milp_val, a.workers == 1,
             os.path.join(adir, f"{tag}{'_nocvar' if a.no_cvar else ''}_{meth}_s{sd}.pt"), betas, a.delay_metric,
             train_kw)
            for meth, sd in jobs]
    if a.workers > 1 and len(args) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_one, arg) for arg in args]
            for f in as_completed(futs):             # write each run as soon as it finishes
                run = f.result(); results["runs"].append(run); report(run)
                json.dump(results, open(out_path, "w"))
    else:
        for arg in args:
            run = run_one(arg); results["runs"].append(run); report(run)
            json.dump(results, open(out_path, "w"))
    print("wrote", out_path)


def report(run):
    ev = run["eval"]
    print(f"== {run['method']} seed {run['seed']}: ret {ev['ret']:.3f} infeas {ev['infeas_rate']:.2f} "
          f"covgap {ev['cov_gap_milp']:.3f} cvar {ev['cvar_grid']:.2f} "
          f"cmdp_ok {int(ev.get('cmdp_ok', -1))} max g/beta {ev.get('cmdp_max_ratio', float('nan')):.2f}", flush=True)


def run_one(arg):
    inst, fpath, meth, sd, iters, eps, use_cvar, ref, milp_val, verbose, actor_path, betas, delay_metric, train_kw = arg
    torch.set_num_threads(1)
    follower = FollowerPolicy(); follower.load_state_dict(torch.load(fpath))
    env = LeaderEnv(inst, follower, seed=sd, delay_metric=delay_metric)
    train_kw = dict(train_kw)
    layer_kw = train_kw.pop("layer_kw", {})
    cfg = LeaderConfig(layer=meth, iters=iters, episodes_per_iter=eps, seed=sd, use_cvar=use_cvar,
                       layer_kw=layer_kw, **betas, **train_kw)
    agent = LeaderAgent(env, cfg, ref)
    t1 = time.time()
    log = agent.train(verbose=verbose, log_every=max(1, iters // 5))
    ev = agent.evaluate()
    ev["cov_gap_milp"] = 1.0 - ev["cov_disc"] / milp_val if milp_val > 0 else float("nan")
    torch.save(agent.actor.state_dict(), actor_path)      # allows re-evaluation without retraining
    return {"method": meth, "seed": sd, "leader_iters": iters, "leader_eps": eps,
            "train_time": time.time() - t1,
            "time_per_update": (time.time() - t1) / iters, "eval": ev, "log": log, "mu": agent.mu}


if __name__ == "__main__":
    main()
