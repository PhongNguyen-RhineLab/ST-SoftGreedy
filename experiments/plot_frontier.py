"""Return versus constraint tightness: heuristic phi-sweeps and learned policies.

x = max_h g_h / beta_h over the active CMDP constraints (<= 1 means CMDP-feasible)
y = discounted return of the deployed plan
Gray lines: the two heuristic families (coverage greedy, greedy + BESS) as the
budget fraction phi goes 0 -> 1. Colored markers: one per training seed.
Hollow markers: the seed deploys an infeasible action at some stage.

  python experiments/plot_frontier.py --json results_m16/exp3_m16_th5_T3_i0.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-4 of the validated reference palette (light mode), fixed order;
# identity is also carried by marker shape, so color is never the only cue
SERIES = [("stsg", r"STSG (ours)", "#2a78d6", "o"),
          ("perturbed", "Perturbed optimiser", "#eb6834", "s"),
          ("rand_greedy", "Randomised greedy", "#1baf7a", "^"),
          ("qp_round", "QP projection + round", "#eda100", "D")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def max_ratio(e, use_cvar):
    keys = ["g_grid", "g_delay"] + (["g_cvar"] if use_cvar else [])
    return max(e[k] for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.json))
    use_cvar = "nocvar" not in os.path.basename(a.json)
    out = a.out or a.json.replace(".json", "_frontier.pdf")

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(1.0, 1.01, "CMDP boundary", color=MUTED, fontsize=6, ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), clip_on=False)

    sweep = d.get("phi_sweep", [])
    for fam, ls, dy in (("greedy", "-", 5), ("greedy+BESS", (0, (1, 1.2)), -12)):
        pts = sorted([e for e in sweep if e["family"] == fam and e.get("n_built", 1) > 0], key=lambda e: e["phi"])
        if not pts:
            continue
        x = [max_ratio(e, use_cvar) for e in pts]; y = [e["ret"] for e in pts]
        ax.plot(x, y, ls=ls, color=MUTED, lw=1.0, marker=".", ms=3, zorder=2)
        ax.annotate(f"{fam}, $\\phi$=1", (x[-1], y[-1]), xytext=(5, dy), textcoords="offset points",
                    fontsize=6, color=MUTED, va="center")

    runs = d["runs"]
    for meth, label, col, mk in SERIES:
        rr = [r for r in runs if r["method"] == meth and "g_grid" in r["eval"]]
        if not rr:
            continue
        x = np.array([max_ratio(r["eval"], use_cvar) for r in rr])
        y = np.array([r["eval"]["ret"] for r in rr])
        feas = np.array([r["eval"]["infeas_rate"] == 0 for r in rr])
        ax.scatter(x[feas], y[feas], s=18, marker=mk, color=col, edgecolor="white", lw=0.6,
                   zorder=4, label=label)
        if (~feas).any():
            ax.scatter(x[~feas], y[~feas], s=18, marker=mk, facecolor="none", edgecolor=col, lw=1.0, zorder=4)
        # seeds that converge to the same plan sit on one point: say how many
        pts = {}
        for xi, yi in zip(np.round(x, 3), np.round(y, 3)):
            pts[(xi, yi)] = pts.get((xi, yi), 0) + 1
        for (xi, yi), k in pts.items():
            if k > 1:
                ax.annotate(f"\u00d7{k}", (xi, yi), xytext=(5, 3), textcoords="offset points",
                            fontsize=6, color=INK)
    empty = [r for r in runs if r["method"] in ("penalty", "lagrangian") and r["eval"]["ret"] == 0]
    if empty:
        ax.scatter([0], [0], s=22, marker="x", color=INK, lw=1.0, zorder=5)
        ax.annotate("penalty, Lagrangian\n(empty plan, all seeds)", (0, 0), xytext=(8, 14),
                    textcoords="offset points", fontsize=6, color=INK)

    ax.set_xlabel(r"max constraint ratio $g/\beta$ (feasible $\leq 1$)")
    ax.set_ylabel("discounted return")
    ax.set_xlim(left=-0.05); ax.set_ylim(bottom=-0.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(right=max(ax.get_xlim()[1], 1.3) * 1.12)          # room for the end-of-line labels
    ax.legend(frameon=False, fontsize=6, loc="upper right", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
