"""Aggregate results/ into LaTeX (results/tables.tex).

tab_main    : exp3, columns Viol. rate / Coverage gap to MILP / Return / CVaR grid / s per update
tab_static  : exp2, coverage gap (mean ± std over seeds) per (m, theta), deployment infeasible rate
tab_abl     : exp3 ablation runs if present
tab_theory  : exp1 summary (bound violations must be 0; error at smallest tau)
Infeasible deployments are excluded from gap averages and flagged with the
infeasible rate, because a gap computed on an infeasible set is not comparable.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

NAMES = {"penalty": "Penalty", "lagrangian": "Lagrangian CMDP", "qp_round": "QP projection + round",
         "gumbel_topk": "Gumbel top-$k$", "rand_greedy": "Randomised greedy",
         "perturbed": "Perturbed optimiser", "stsg": r"\STSG{} (ours)",
         "stsg_density": r"\STSG{}, density key", "stsg_softcost": r"\STSG{}, soft-sorted costs",
         "abl_no_matroid_gate": "(a) no matroid gate", "abl_naive_softsort": "(b) naive soft-sort",
         "abl_soft_forward": "(c) soft forward", "abl_paper_gate_offset": "gate offset 1 (paper text)",
         "greedy_gain": "Greedy, re-ranked gain", "greedy_density": "Greedy, re-ranked density",
         "greedy_best": "Greedy, best of both", "static_key": "Alg. 1, static modular key"}


def ms(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "---"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f}$\\pm${v.std(ddof=1):.3f}"


def tab_main(files, methods, caption, label):
    rows = defaultdict(list)
    for f in files:
        for r in json.load(open(f))["runs"]:
            rows[r["method"]].append(r)
    if not any(m in rows for m in methods):
        return ""
    out = [r"\begin{table}[t]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
           r"\begin{tabular}{lccccc}\toprule",
           r"Method & Viol.\ rate & Cov.\ gap & Return & CVaR$_\alpha$ grid & s/update \\ \midrule"]
    for m in methods:
        if m not in rows:
            continue
        rr = rows[m]
        ev = [r["eval"] for r in rr]
        feas = [e for e in ev if e["infeas_rate"] == 0]
        out.append(f"{NAMES.get(m, m)} & {ms([e['infeas_rate'] for e in ev])} & "
                   f"{ms([e['cov_gap_milp'] for e in feas]) if feas else '---'} & "
                   f"{ms([e['ret'] for e in ev])} & {ms([e['cvar_grid'] for e in ev])} & "
                   f"{ms([r['time_per_update'] for r in rr])} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def tab_static(path):
    if not os.path.exists(path):
        return ""
    recs = json.load(open(path))
    groups = sorted({(r["m"], r["theta"]) for r in recs})
    methods = list(dict.fromkeys(r["method"] for r in recs))
    head = " & ".join(f"$m{{=}}{m},\\theta{{=}}{t:g}$" for m, t in groups)
    out = [r"\begin{table*}[t]\centering\scriptsize",
           r"\caption{Static coverage: gap to MILP on feasible deployments (mean$\pm$sd over seeds); "
           r"superscript = fraction of infeasible deployments when non-zero.}\label{tab:static}",
           r"\begin{tabular}{l" + "c" * len(groups) + r"}\toprule", f"Method & {head} \\\\ \\midrule"]
    for meth in methods:
        cells = []
        for g in groups:
            rr = [r for r in recs if r["method"] == meth and (r["m"], r["theta"]) == g]
            inf = np.mean([not r["feasible"] for r in rr])
            gap = ms([r["gap"] for r in rr if r["feasible"]])
            cells.append(gap + (f"$^{{{inf:.2f}}}$" if inf > 0 else ""))
        out.append(f"{NAMES.get(meth, meth)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule\end{tabular}\end{table*}", ""]
    return "\n".join(out)


def tab_theory(path):
    if not os.path.exists(path):
        return ""
    rows = list(csv.DictReader(open(path)))
    out = [r"\begin{table}[t]\centering\small",
           r"\caption{Consistency check (Exp.~1) at the smallest temperature.}\label{tab:theory}",
           r"\begin{tabular}{lcccccc}\toprule",
           r"$N$ & $\theta$ & offset & $\|\tilde x-\bar x\|_1$ & precond. & bound viol. & rank viol. \\ \midrule"]
    by = defaultdict(list)
    for r in rows:
        by[(int(r["m"]), float(r["theta"]), float(r["gate_offset"]))].append(r)
    for (m, th, off), rr in sorted(by.items()):
        r = min(rr, key=lambda z: float(z["tau"]))
        out.append(f"{4*m} & {th:g} & {off:g} & {float(r['err_mean']):.3g} & {float(r['precond_frac']):.2f} & "
                   f"{r['bound_violations']} & {r['rank_bound_violations']} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    main_files = [f for f in glob.glob(os.path.join(a.res, "exp3_*.json")) if "nocvar" not in f]
    tex = [r"% generated by experiments/make_tables.py",
           tab_main(main_files, ["penalty", "lagrangian", "qp_round", "gumbel_topk", "rand_greedy",
                                 "perturbed", "stsg"],
                    "Main results on the MCS--BSS corridor. Viol.\\ rate: fraction of stages with an "
                    "infeasible deployed action. Cov.\\ gap: discounted coverage gap to the multistage MILP, "
                    "feasible runs only; the MILP does not model the follower game.", "tab:main"),
           tab_main(main_files, ["stsg", "abl_no_matroid_gate", "abl_naive_softsort", "abl_soft_forward",
                                 "stsg_density", "stsg_softcost"], "Ablations (a)--(d).", "tab:abl"),
           tab_static(os.path.join(a.res, "exp2_static.json")),
           tab_theory(os.path.join(a.res, "exp1_consistency.csv"))]
    path = os.path.join(a.res, "tables.tex")
    open(path, "w").write("\n".join(t for t in tex if t))
    print("wrote", path)


if __name__ == "__main__":
    main()
