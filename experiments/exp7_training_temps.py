"""Exp 7: the consistency theory and the straight-through bias at the temperatures
actually used in training (tau1 = 0.1, tau2 = 0.05), next to a temperature sweep.

Part A  constraint systems from the corridor generator with Gaussian scores
        (as Exp 1), N in {32, 64, 128}.
Part B  states and scores visited by the TRAINED leaders (optional; needs the
        results directory of exp3 with actors/ and the follower checkpoint).
For every system we report, at each (tau1, tau2):
  gamma, delta                  score gap and budget margin (Definition, corrected)
  cond_i, cond_ii, precond      hypotheses of the corrected gate lemma
  bound, err                    Theorem bound and measured ||x_tilde - x_bar||_1
  cos_st, relerr_st             straight-through gradient  g_ST = (dx~/dy)^T grad J(x_bar)
                                vs surrogate gradient      g_S  = (dx~/dy)^T grad J(x_tilde)
                                with J = probabilistic coverage sum_d w_d (1 - exp(-kappa (A x)_d)),
                                a smooth monotone submodular extension with Lipschitz gradient
                                (the clamp-based coverage is piecewise LINEAR, so g_ST = g_S
                                trivially there and the comparison would be vacuous)
  gnorm                         ||g_S||_2, to show how the useful signal scales with tau

  python experiments/exp7_training_temps.py --out results
  python experiments/exp7_training_temps.py --out results_m16 --actors results_m16 --m_actor 16
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

from env.instance import generate, random_feasible_config

KAPPA = 3.0


def smooth_cov(x, A, w):
    return (w * (1 - torch.exp(-KAPPA * (x @ A.T)))).sum(-1)
from stsg.layer import STSG, ConstraintBatch, consistency_diagnostics

TRAIN = (0.1, 0.05)
SWEEP = [(t, t) for t in (1.0, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001)] + [TRAIN]


def st_bias(y, cb, A, w, tau1, tau2):
    """Per-row cosine and relative error between g_ST and g_S, and ||g_S||."""
    layer = STSG(tau1, tau2)
    y1 = y.clone().requires_grad_(True)
    out = layer(y1, cb)
    J = lambda x: torch.stack([smooth_cov(x[b:b + 1], A[b], w[b]) for b in range(x.shape[0])]).sum()
    g_st, = torch.autograd.grad(J(out.x), y1)
    y2 = y.clone().requires_grad_(True)
    x_tilde, _ = layer.relaxed(y2, cb)
    g_s, = torch.autograd.grad(J(x_tilde), y2)
    num = (g_st * g_s).sum(-1)
    den = g_st.norm(dim=-1) * g_s.norm(dim=-1)
    cos = torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.full_like(num, float("nan")))
    rel = (g_st - g_s).norm(dim=-1) / g_s.norm(dim=-1).clamp_min(1e-12)
    rel = torch.where(g_s.norm(dim=-1) > 1e-12, rel, torch.full_like(rel, float("nan")))
    return cos, rel, g_s.norm(dim=-1), g_st.norm(dim=-1)


def generator_systems(m, theta, B, seed, score_scale):
    rng = np.random.default_rng(seed)
    cbs, As, ws = [], [], []
    for b in range(B):
        inst = generate(m=m, theta=theta, seed=seed * 1000 + b)
        t = int(rng.integers(inst.T))
        built = random_feasible_config(inst, t - 1, rng) if t > 0 else np.zeros(inst.N)
        cbs.append(inst.constraint_batch(built, t))
        As.append(torch.as_tensor(inst.A, dtype=torch.float32))
        ws.append(torch.as_tensor(inst.w / inst.w.sum(), dtype=torch.float32))
    cb = ConstraintBatch(*(torch.cat([getattr(c, f) for c in cbs]) for f in ("c", "budget", "part", "cap", "avail")))
    y = torch.randn(cb.c.shape, generator=torch.Generator().manual_seed(seed)) * score_scale
    return y, cb, As, ws


def actor_systems(res_dir, m, theta, T):
    """(y, cb) at every stage of the deterministic rollout of every stored STSG actor."""
    from agents.follower import FollowerPolicy
    from agents.leader import ScoreNet
    from env.leader_env import LeaderEnv, stack_states
    tag = f"m{m}_th{theta:g}_T{T}_i0"
    inst = generate(m=m, theta=theta, T=T, seed=0)
    fol = FollowerPolicy(); fol.load_state_dict(torch.load(os.path.join(res_dir, f"follower_{tag}.pt")))
    env = LeaderEnv(inst, fol, seed=0)
    ys, cbs = [], []
    files = sorted(glob.glob(os.path.join(res_dir, "actors", f"{tag}_stsg_s*.pt")))
    for f in files:
        net = ScoreNet(); net.load_state_dict(torch.load(f))
        s = env.reset(); env.k = 0; done = False
        while not done:
            feats, glob_, cb = stack_states([s])
            with torch.no_grad():
                y = net(feats, glob_)
            ys.append(y); cbs.append(cb)
            x = STSG(*TRAIN)(y, cb).deploy[0].numpy()
            s, _, _, done, _ = env.step(x)
    if not ys:
        raise SystemExit(f"no STSG actors found under {res_dir}/actors for tag {tag}")
    cb = ConstraintBatch(*(torch.cat([getattr(c, f) for c in cbs]) for f in ("c", "budget", "part", "cap", "avail")))
    A = torch.as_tensor(inst.A, dtype=torch.float32); w = torch.as_tensor(inst.w / inst.w.sum(), dtype=torch.float32)
    return torch.cat(ys), cb, [A] * len(ys), [w] * len(ys), len(files)


def summarise(tag, y, cb, As, ws, rows, extra):
    for tau1, tau2 in SWEEP:
        d = consistency_diagnostics(y, cb, tau1, tau2)
        cos, rel, gn, gst = st_bias(y, cb, As, ws, tau1, tau2)
        ok = d["precond_ok"]
        f = lambda v: float(torch.nanmedian(v)) if v.numel() else float("nan")
        rows.append(dict(source=tag, tau1=tau1, tau2=tau2, train=(tau1, tau2) == TRAIN, n=len(y), **extra,
                         precond=float(ok.float().mean()), cond_i=float(d["cond_i"].float().mean()),
                         cond_ii=float(d["cond_ii"].float().mean()),
                         gamma_med=f(d["gamma"]), delta_med=f(d["delta"]),
                         bound_med=f(d["bound"]), err_mean=float(d["err_l1"].mean()), err_med=f(d["err_l1"]),
                         bound_viol=int((d["err_l1"][ok] > d["bound"][ok] + 1e-6).sum()),
                         cos_st_med=f(cos), relerr_st_med=f(rel), gnorm_s_med=f(gn), gnorm_st_med=f(gst)))
        r = rows[-1]
        print(f"[{tag}{'' if not extra else ' ' + str(extra)}] tau=({tau1},{tau2}){' TRAIN' if r['train'] else ''}: "
              f"precond {r['precond']:.2f} (i {r['cond_i']:.2f}, ii {r['cond_ii']:.2f}) "
              f"gamma {r['gamma_med']:.3g} delta {r['delta_med']:.3g} bound {r['bound_med']:.3g} "
              f"err {r['err_mean']:.3g} | cos(g_ST,g_S) {r['cos_st_med']:.3f} relerr {r['relerr_st_med']:.3f} "
              f"||g_S|| {r['gnorm_s_med']:.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--score_scale", type=float, default=3.0)
    ap.add_argument("--actors", default=None, help="exp3 results dir with actors/ (Part B)")
    ap.add_argument("--m_actor", type=int, default=16)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--skip_generator", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    if not a.skip_generator:
        for m in a.ms:
            y, cb, As, ws = generator_systems(m, a.theta, a.B, seed=100 + m, score_scale=a.score_scale)
            summarise("generator", y, cb, As, ws, rows, {"N": 4 * m})
    if a.actors:
        y, cb, As, ws, n_act = actor_systems(a.actors, a.m_actor, a.theta, a.T)
        summarise("trained_actors", y, cb, As, ws, rows, {"N": 4 * a.m_actor, "actors": n_act})
    path = os.path.join(a.out, "exp7_training_temps" + ("_actors" if a.actors else "") + ".csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader(); wr.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
