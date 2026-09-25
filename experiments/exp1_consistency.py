"""Exp 1 (theory check, no RL). Outputs results/exp1_*.csv and a figure.

(1) ||x_tilde - x_bar||_1 against the Theorem consistency bound, tau1 = tau2 sweep,
    for gate_offset 0.5 (corrected) and 1.0 (paper text).
(2) Lemma rank: ||P^s - P_pi||_inf and column-sum deviation against 2(N-1)e^{-gamma/tau1}.
(3) Ablation b: naive soft-sort error.
(4) Ablation c: soft forward, budget overflow and infeasibility after rounding.
Instances: constraint systems drawn from the corridor generator (random stage,
random partial build) with Gaussian scores.
"""
import argparse
import csv
import os

import numpy as np
import torch

from env.instance import generate, random_feasible_config
from stsg.layer import STSG, ConstraintBatch, consistency_diagnostics


def corridor_batch(m, theta, B, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for b in range(B):
        inst = generate(m=m, theta=theta, seed=seed * 1000 + b)
        t = int(rng.integers(inst.T))
        built = random_feasible_config(inst, t - 1, rng) if t > 0 else np.zeros(inst.N)
        rows.append(inst.constraint_batch(built, t))
    return ConstraintBatch(*(torch.cat([getattr(r, f) for r in rows])
                             for f in ("c", "budget", "part", "cap", "avail")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--thetas", type=float, nargs="+", default=[2, 5, 10])
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--score_scale", type=float, default=3.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    taus = np.logspace(-3.5, 0, 15)
    rows = []
    for m in a.ms:
        for th in a.thetas:
            cb = corridor_batch(m, th, a.B, seed=int(m * 10 + th))
            y = torch.randn(cb.c.shape) * a.score_scale
            for tau in taus:
                for off in (0.5, 1.0):
                    d = consistency_diagnostics(y, cb, tau, tau, gate_offset=off)
                    ok = d["precond_ok"]
                    rows.append(dict(
                        m=m, theta=th, tau=tau, gate_offset=off,
                        err_mean=d["err_l1"].mean().item(), err_max=d["err_l1"].max().item(),
                        bound_median=d["bound"].median().item(),
                        precond_frac=ok.float().mean().item(),
                        bound_violations=int((d["err_l1"][ok] > d["bound"][ok] + 1e-6).sum()),
                        rank_err_max=d["rank_err"].max().item(),
                        rank_bound_violations=int((d["rank_err"] > d["rank_bound"] + 1e-6).sum()),
                        col_dev_max=d["col_dev"].max().item(),
                        delta_median=d["delta"].median().item(),
                        delta_paper_median=d["delta_part_paper"].median().item()))
                # naive softsort and soft forward (offset 0.5)
                with torch.no_grad():
                    xb = STSG(tau, tau)(y, cb).deploy
                    xn, _ = STSG(tau, tau, sort="naive").relaxed(y, cb)
                    soft = STSG(tau, tau, forward_mode="soft")(y, cb)
                    v_soft = cb.violation(soft.x)
                    v_round = cb.violation(soft.deploy)
                extra = dict(naive_err_mean=(xn - xb).abs().sum(-1).mean().item(),
                                soft_budget_overflow=v_soft["budget"].mean().item(),
                                soft_round_infeasible=v_round["infeasible"].mean().item())
                rows[-1].update(extra); rows[-2].update(extra)
            print(f"m={m} theta={th} done")
    path = os.path.join(a.out, "exp1_consistency.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        wr.writeheader(); wr.writerows(rows)
    print("wrote", path)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for m in a.ms:
            sel = [r for r in rows if r["m"] == m and r["theta"] == a.thetas[len(a.thetas) // 2]]
            for off, ls in ((0.5, "-"), (1.0, "--")):
                rr = [r for r in sel if r["gate_offset"] == off]
                ax[0].loglog([r["tau"] for r in rr], [r["err_mean"] + 1e-8 for r in rr], ls,
                             label=f"N={4*m} offset {off}")
            rr = [r for r in sel if r["gate_offset"] == 0.5 and "soft_round_infeasible" in r]
            ax[1].semilogx([r["tau"] for r in rr], [r["soft_round_infeasible"] for r in rr], label=f"N={4*m}")
        ax[0].set_xlabel("tau1 = tau2"); ax[0].set_ylabel("mean ||x_tilde - x_bar||_1"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("tau2"); ax[1].set_ylabel("soft forward: infeasible after rounding"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "exp1_consistency.pdf"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
