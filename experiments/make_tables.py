"""Aggregate results/ into LaTeX (results/tables.tex).

Every cell is  mean [95% percentile-bootstrap CI]  over the unit of
replication (training seed in exp3, instance in exp2), with n reported.

tab:main          exp3 main methods + non-learned reference plans (single deterministic value)
tab:abl           exp3 ablations
tab:paired        exp3, paired by seed: method - STSG on Return, CVaR grid, Viol. rate;
                  Wilcoxon signed-rank p, Holm-adjusted within each metric column
tab:static        exp2 coverage gap, feasible deployments only, blocks per m, columns per theta
tab:static_paired exp2, paired by instance (m, theta, seed), pooled per m and overall;
                  pairs chosen with --static_pairs a:b (diff = gap_a - gap_b, negative = a better)
tab:theory        exp1 summary (unchanged)

Infeasible deployments are excluded from gap statistics (n shows how many
remain), because a gap computed on an infeasible set is not comparable.
Rows with non-zero violation are marked in tab:paired: their Return is not
comparable to a feasible method's Return.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from metrics.stats import boot_ci, paired_compare, holm, fmt_ci, fmt_p, fnum

NAMES = {"penalty": "Penalty", "lagrangian": "Lagrangian CMDP", "qp_round": "QP projection + round",
         "gumbel_topk": "Gumbel top-$k$", "rand_greedy": "Randomised greedy",
         "perturbed": "Perturbed optimiser", "stsg": r"\STSG{} (ours)",
         "stsg_density": r"\STSG{}, density key", "stsg_softcost": r"\STSG{}, soft-sorted costs",
         "abl_no_matroid_gate": "(a) no matroid gate", "abl_naive_softsort": "(b) naive soft-sort",
         "abl_soft_forward": "(c) soft forward", "abl_paper_gate_offset": "gate offset 1 (paper text)",
         "greedy_gain": "Greedy, re-ranked gain", "greedy_density": "Greedy, re-ranked density",
         "greedy_best": "Greedy, best of both", "static_key": "Alg.~1, static modular key",
         "ref_greedy_rerank": r"Ref.: re-ranked coverage greedy$^\dagger$",
         "ref_static_key": r"Ref.: Alg.~1, static key$^\dagger$",
         "ref_greedy_constrained": r"Ref.: best CMDP-feasible heuristic$^\dagger$"}
MAIN = ["penalty", "lagrangian", "qp_round", "gumbel_topk", "rand_greedy", "perturbed", "stsg"]
ABL = ["stsg", "abl_no_matroid_gate", "abl_naive_softsort", "abl_soft_forward", "stsg_density", "stsg_softcost"]
DEFAULT_STATIC_PAIRS = ["stsg:static_key", "stsg:abl_no_matroid_gate", "stsg:abl_paper_gate_offset",
                        "stsg:abl_naive_softsort", "stsg_softcost:stsg", "stsg_softcost:rand_greedy",
                        "stsg_softcost:perturbed", "stsg_softcost:greedy_best"]


def ci_cell(v, digits=3):
    m, lo, hi, n = boot_ci(v)
    return fmt_ci(m, lo, hi, n, digits)


# ---------------------------------------------------------------- exp3
def load_exp3(files):
    runs, refs = defaultdict(dict), {}
    for f in files:
        d = json.load(open(f))
        for r in d["runs"]:
            runs[r["method"]][r["seed"]] = r            # later file wins on duplicates
        refs.update(d.get("references", {}))
    return runs, refs


def tab_exp3(runs, refs, methods, caption, label, with_refs):
    if not any(m in runs for m in methods):
        return ""
    out = [r"\begin{table*}[t]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
           r"\begin{tabular}{lcccccccc}\toprule",
           r"Method & $n$ & Viol.\ rate & CMDP-ok & Built & Cov.\ gap$^\ast$ & Return & CVaR$_\alpha$ grid & s/update \\ \midrule"]
    if with_refs:
        for name, ev in refs.items():
            ok = ("yes" if ev["cmdp_ok"] else "no") if "cmdp_ok" in ev else "---"
            phi = f" ({ev.get('family', 'greedy')}, $\\phi={ev['phi']:g}$)" if "phi" in ev else ""
            out.append(f"{NAMES.get(name, name)}{phi} & -- & {fnum(ev['infeas_rate'])} & {ok} & "
                       f"{int(sum(ev['built_example']))} & "
                       f"{fnum(ev['cov_gap_milp'])} & {fnum(ev['ret'])} & {fnum(ev['cvar_grid'])} & -- \\\\")
        if refs:
            out.append(r"\midrule")
    for m in methods:
        if m not in runs:
            continue
        rr = list(runs[m].values())
        ev = [r["eval"] for r in rr]
        feas = [e["cov_gap_milp"] for e in ev if e["infeas_rate"] == 0]
        gap = ci_cell(feas) + (f" ({len(feas)})" if len(feas) < len(ev) else "")
        out.append(f"{NAMES.get(m, m)} & {len(rr)} & {ci_cell([e['infeas_rate'] for e in ev])} & "
                   f"{count_ok(ev)} & {ci_cell([sum(e['built_example']) for e in ev], 1)} & {gap} & "
                   f"{ci_cell([e['ret'] for e in ev])} & {ci_cell([e['cvar_grid'] for e in ev])} & "
                   f"{ci_cell([r['time_per_update'] for r in rr], 2)} \\\\")
    out += [r"\bottomrule\end{tabular}",
            r"\\[2pt]\footnotesize Cells: mean [95\% bootstrap CI] over training seeds. "
            r"$^\ast$feasible runs only, count in parentheses when some runs are infeasible. "
            r"$^\dagger$deterministic plan, one value. Built: modules installed after the last stage (0 = empty plan, "
            r"which satisfies every constraint trivially). CMDP-ok: seeds whose deployed policy satisfies every "
            r"constraint of the CMDP and is feasible; details in Table~\ref{tab:constraints}.",
            r"\end{table*}", ""]
    return "\n".join(out)


def count_ok(ev):
    have = [e for e in ev if "cmdp_ok" in e]
    if not have:
        return "---"
    return f"{int(sum(e['cmdp_ok'] for e in have))}/{len(have)}"


def tab_constraints(runs, refs, methods, label="tab:constraints"):
    rows = [m for m in methods if m in runs and any("g_grid" in r["eval"] for r in runs[m].values())]
    rref = {k: v for k, v in refs.items() if "g_grid" in v}
    if not rows and not rref:
        return ""
    out = [r"\begin{table*}[t]\centering\small",
           r"\caption{CMDP constraints at deployment, as ratios $g/\beta$ (a value $\le 1$ means the constraint "
           r"holds): discounted expected grid and delay cost and CVaR$_\alpha$ of episode grid cost, each normalised "
           r"by the calibration plan. Mean [95\% bootstrap CI] over seeds; CMDP-ok counts seeds satisfying all "
           r"constraints with a feasible action.}" + f"\\label{{{label}}}",
           r"\begin{tabular}{lcccccc}\toprule",
           r"Method & $n$ & grid $g/\beta$ & delay $g/\beta$ & CVaR $g/\beta$ & max $g/\beta$ & CMDP-ok \\ \midrule"]
    for name, ev in rref.items():
        phi = f" ({ev.get('family', 'greedy')}, $\\phi={ev['phi']:g}$)" if "phi" in ev else ""
        out.append(f"{NAMES.get(name, name)}{phi} & -- & {fnum(ev['g_grid'])} & {fnum(ev['g_delay'])} & "
                   f"{fnum(ev['g_cvar'])} & {fnum(ev['cmdp_max_ratio'])} & {'yes' if ev['cmdp_ok'] else 'no'} \\\\")
    if rref and rows:
        out.append(r"\midrule")
    for m in rows:
        ev = [r["eval"] for r in runs[m].values() if "g_grid" in r["eval"]]
        out.append(f"{NAMES.get(m, m)} & {len(ev)} & " + " & ".join(
            ci_cell([e[k] for e in ev]) for k in ("g_grid", "g_delay", "g_cvar", "cmdp_max_ratio"))
            + f" & {count_ok(ev)} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table*}", ""]
    return "\n".join(out)


def tab_paired_exp3(runs, methods, base="stsg"):
    if base not in runs:
        return ""
    metrics = [("ret", "Return", +1), ("cvar_grid", "CVaR$_\\alpha$ grid", -1)]
    others = [m for m in methods if m != base and m in runs]
    if not others:
        return ""
    cmp = {}
    for key, _, _ in metrics:
        b = {s: r["eval"][key] for s, r in runs[base].items()}
        res = [paired_compare({s: r["eval"][key] for s, r in runs[m].items()}, b) for m in others]
        adj = holm([r["p"] for r in res])
        for m, r, pa in zip(others, res, adj):
            cmp[(m, key)] = (r, pa)
    out = [r"\begin{table*}[t]\centering\small",
           f"\\caption{{Paired comparison against {NAMES[base]} over matched training seeds: "
           r"mean difference (method $-$ \STSG{}) [95\% bootstrap CI], Wilcoxon signed-rank $p$ "
           r"Holm-adjusted within each metric. Return: higher is better; CVaR and violation: lower is better. "
           r"$^\ddagger$method deploys infeasible actions, so its Return is not comparable. "
           r"Viol.\ seeds: seeds with at least one infeasible stage; no test is reported because \STSG{} is "
           r"feasible by construction (Proposition~\ref{prop:feasible}), so every non-zero difference has the same "
           r"sign and the Wilcoxon test only counts those seeds.}\label{tab:paired}",
           r"\begin{tabular}{lc" + "cc" * len(metrics) + r"cc}\toprule",
           "Method & $n$ & " + " & ".join(f"$\\Delta$ {lab} & $p_{{\\mathrm{{Holm}}}}$" for _, lab, _ in metrics)
           + r" & Viol.\ seeds & $\Delta$ Viol.\ rate \\ \midrule"]
    for m in others:
        infeas = np.mean([r["eval"]["infeas_rate"] for r in runs[m].values()]) > 0
        n = cmp[(m, "ret")][0]["n"]
        cells = []
        for key, _, _ in metrics:
            r, pa = cmp[(m, key)]
            cells += [fmt_ci(r["diff"], r["lo"], r["hi"], r["n"]), fmt_p(pa)]
        vio = [r["eval"]["infeas_rate"] for r in runs[m].values()]
        dv = paired_compare({s_: r["eval"]["infeas_rate"] for s_, r in runs[m].items()},
                            {s_: r["eval"]["infeas_rate"] for s_, r in runs[base].items()})
        cells += [f"{sum(v > 0 for v in vio)}/{len(vio)}", fmt_ci(dv["diff"], dv["lo"], dv["hi"], dv["n"])]
        mark = r"$^\ddagger$" if infeas else ""                 # no backslash inside f-string braces (py<3.12)
        out.append(f"{NAMES.get(m, m)}{mark} & {n} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule\end{tabular}\end{table*}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- exp2
def tab_static(recs):
    ms = sorted({r["m"] for r in recs}); ths = sorted({r["theta"] for r in recs})
    methods = list(dict.fromkeys(r["method"] for r in recs))
    out = [r"\begin{table*}[t]\centering\scriptsize",
           r"\caption{Static coverage: gap to MILP, mean [95\% bootstrap CI] over instances, feasible deployments "
           r"only; superscript = fraction of infeasible deployments when non-zero.}\label{tab:static}",
           r"\begin{tabular}{l" + "c" * len(ths) + r"}\toprule",
           "Method & " + " & ".join(f"$\\theta={t:g}$" for t in ths) + r" \\"]
    for m in ms:
        out.append(r"\midrule" + f"\\multicolumn{{{len(ths) + 1}}}{{l}}{{$m={m}$, $N={4 * m}$}} \\\\ \\midrule")
        for meth in methods:
            cells = []
            for t in ths:
                rr = [r for r in recs if r["method"] == meth and r["m"] == m and r["theta"] == t]
                if not rr:
                    cells.append("---"); continue
                inf = np.mean([not r["feasible"] for r in rr])
                cells.append(ci_cell([r["gap"] for r in rr if r["feasible"]]) +
                             (f"$^{{{inf:.2f}}}$" if inf > 0 else ""))
            out.append(f"{NAMES.get(meth, meth)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule\end{tabular}\end{table*}", ""]
    return "\n".join(out)


def tab_static_paired(recs, pairs):
    ms = sorted({r["m"] for r in recs})
    have = {r["method"] for r in recs}
    pairs = [p.split(":") for p in pairs]
    pairs = [(a, b) for a, b in pairs if a in have and b in have]
    if not pairs:
        return ""

    def unit_gaps(meth, m=None):
        return {(r["m"], r["theta"], r["seed"]): r["gap"] for r in recs
                if r["method"] == meth and r["feasible"] and (m is None or r["m"] == m)}

    scopes = [(f"$m={m}$", m) for m in ms] + [("all", None)]
    res = {(a, b, lab): paired_compare(unit_gaps(a, m), unit_gaps(b, m)) for a, b in pairs for lab, m in scopes}
    for lab, _ in scopes:                                   # Holm within each scope
        adj = holm([res[(a, b, lab)]["p"] for a, b in pairs])
        for (a, b), pa in zip(pairs, adj):
            res[(a, b, lab)]["p_holm"] = pa
    out = [r"\begin{table*}[t]\centering\scriptsize",
           r"\caption{Static coverage, paired by instance: $\Delta$ = gap(A) $-$ gap(B), negative means A is "
           r"better; [95\% bootstrap CI]; $n$ = instances where both deploy feasibly; Wilcoxon $p$ Holm-adjusted "
           r"within each column block.}\label{tab:static_paired}",
           r"\begin{tabular}{ll" + "ccc" * len(scopes) + r"}\toprule",
           "A & B & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for lab, _ in scopes) + r" \\",
           " & & " + " & ".join(r"$n$ & $\Delta$ & $p$" for _ in scopes) + r" \\ \midrule"]
    for a, b in pairs:
        cells = []
        for lab, _ in scopes:
            r = res[(a, b, lab)]
            cells += [str(r["n"]), fmt_ci(r["diff"], r["lo"], r["hi"], r["n"]), fmt_p(r["p_holm"])]
        out.append(f"{NAMES.get(a, a)} & {NAMES.get(b, b)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule\end{tabular}\end{table*}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- exp1
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
    ap.add_argument("--static_pairs", nargs="+", default=DEFAULT_STATIC_PAIRS,
                    help="A:B pairs for tab:static_paired")
    a = ap.parse_args()
    tex = [r"% generated by experiments/make_tables.py  (needs \usepackage{booktabs})"]
    files = sorted(f for f in glob.glob(os.path.join(a.res, "exp3_*.json")) if "nocvar" not in f)
    if files:
        runs, refs = load_exp3(files)
        tex += [tab_exp3(runs, refs, MAIN,
                         "Main results on the MCS--BSS corridor. Viol.\\ rate: fraction of stages with an "
                         "infeasible deployed action. Cov.\\ gap: discounted coverage gap to the multistage "
                         "MILP; the MILP does not model the follower game.", "tab:main", True),
                tab_constraints(runs, refs, MAIN),
                tab_paired_exp3(runs, MAIN),
                tab_exp3(runs, refs, ABL, "Ablations (a)--(d).", "tab:abl", False)]
        abl_p = tab_paired_exp3(runs, ABL)
        tex.append(abl_p.replace(r"\label{tab:paired}", r"\label{tab:paired_abl}"))
    nocvar = sorted(glob.glob(os.path.join(a.res, "exp3_*_nocvar.json")))
    if nocvar:
        runs_e, _ = load_exp3(nocvar)
        tex.append(tab_exp3(runs_e, {}, ["stsg"], "Ablation (e): without the CVaR constraint.",
                            "tab:abl_e", False))
    p2 = os.path.join(a.res, "exp2_static.json")
    if os.path.exists(p2):
        recs = json.load(open(p2))
        tex += [tab_static(recs), tab_static_paired(recs, a.static_pairs)]
    tex.append(tab_theory(os.path.join(a.res, "exp1_consistency.csv")))
    path = os.path.join(a.res, "tables.tex")
    open(path, "w").write("\n".join(t for t in tex if t))
    print("wrote", path)


if __name__ == "__main__":
    main()
