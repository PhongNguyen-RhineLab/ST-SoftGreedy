"""Seed-level statistics for the tables.

boot_ci          percentile bootstrap CI of the mean (unit = one seed / one instance)
paired_compare   per-unit differences d_s = a_s - b_s on units present in both:
                   mean diff, bootstrap CI of the mean diff, Wilcoxon signed-rank p
holm             Holm step-down adjustment over a family of comparisons

Caveats that belong in the paper:
  * with n seeds the percentile bootstrap under-covers for small n; n < 5 is
    reported but should not be interpreted.
  * the exact two-sided Wilcoxon p cannot go below 2 / 2^n, so n <= 5 can
    never reach p < 0.05. That is a property of the test, not evidence of no effect.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

B_BOOT = 10_000


def _clean(v):
    v = np.asarray(v, dtype=float)
    return v[np.isfinite(v)]


def boot_ci(v, alpha=0.05, B=B_BOOT, seed=0):
    """(mean, lo, hi, n). lo = hi = mean when n == 1; all nan when n == 0."""
    v = _clean(v)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if n == 1 or np.ptp(v) == 0:
        return m, m, m, n
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(B, n))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return m, float(lo), float(hi), n


def paired_compare(a: dict, b: dict, alpha=0.05, B=B_BOOT, seed=0):
    """a, b: {unit_id: value}. Returns dict with n, mean diff a-b, CI, Wilcoxon p."""
    keys = sorted(k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k]))
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    n = len(d)
    out = {"n": n, "diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    if n == 0:
        return out
    out["diff"], out["lo"], out["hi"], _ = boot_ci(d, alpha, B, seed)
    if n >= 2 and np.any(d != 0):
        # exact distribution for small n; zero differences dropped (Wilcoxon's convention)
        out["p"] = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                                  method="exact" if n <= 25 else "approx").pvalue)
    elif n >= 1:
        out["p"] = 1.0
    return out


def holm(pvals):
    """Holm-adjusted p-values, same order as input; nan stays nan."""
    p = np.asarray(pvals, dtype=float)
    idx = [i for i in range(len(p)) if np.isfinite(p[i])]
    adj = np.full(len(p), np.nan)
    order = sorted(idx, key=lambda i: p[i])
    m = len(order)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[i]))
        adj[i] = running
    return adj.tolist()


def fnum(x, digits=3):
    """Fixed-point with signed zero suppressed (no '-0.000')."""
    if not np.isfinite(x):
        return "---"
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    return f"{x:.{digits}f}"


def fmt_ci(mean, lo, hi, n, digits=3):
    if n == 0 or not np.isfinite(mean):
        return "---"
    s = fnum(mean, digits)
    if n == 1 or lo == hi:
        return s
    return f"{s} [{fnum(lo, digits)}, {fnum(hi, digits)}]"


def fmt_p(p):
    if not np.isfinite(p):
        return "---"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"
