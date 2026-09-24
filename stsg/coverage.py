"""Corridor coverage f(S) = sum_d w_d * min(1, sum_{j in S} A_dj).

Concave-of-modular with nonnegative A, hence monotone submodular with
f(empty) = 0 (the setting of Theorem transfer). It is also MILP-linearisable,
which gives the exact reference of the Experimental Protocol.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy import sparse


def coverage_torch(x: torch.Tensor, A: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x (B,N) possibly fractional -> (B,). Continuous extension used for pathwise grads."""
    return (w * torch.clamp(x @ A.T, max=1.0)).sum(-1)


def coverage_np(x: np.ndarray, A: np.ndarray, w: np.ndarray) -> float:
    return float((w * np.minimum(1.0, A @ x)).sum())


def feasible_np(x, c, budget, part, cap, avail, tol=1e-9) -> bool:
    if (x * (1 - avail)).sum() > tol:
        return False
    if (c * x).sum() > budget + tol:
        return False
    load = np.bincount(part, weights=x, minlength=len(cap))
    return bool((load <= cap + tol).all())


def rerank_greedy(A, w, c, budget, part, cap, avail, base=None, rule="gain"):
    """Classical greedy with marginal gains recomputed after every pick.

    rule "gain": argmax Δf ; rule "density": argmax Δf / c ; "best": better of both
    (the modified greedy for knapsack). This is the greedy that the p-system
    analysis actually covers; Algorithm 1 ranks once by a static key.
    """
    if rule == "best":
        a = rerank_greedy(A, w, c, budget, part, cap, avail, base, "gain")
        b = rerank_greedy(A, w, c, budget, part, cap, avail, base, "density")
        return a if coverage_np(a, A, w) >= coverage_np(b, A, w) else b
    N = len(c)
    x = np.zeros(N) if base is None else base.copy()
    new = np.zeros(N)
    b = float(budget); n = cap.astype(float).copy()
    cur = np.minimum(1.0, A @ x)
    while True:
        feas = (avail > 0) & (new == 0) & (x == 0) & (c <= b + 1e-9) & (n[part] >= 1)
        if not feas.any():
            break
        gain = (w[:, None] * (np.minimum(1.0, cur[:, None] + A) - cur[:, None])).sum(0)
        score = gain if rule == "gain" else gain / c
        score = np.where(feas, score, -np.inf)
        i = int(np.argmax(score))
        if gain[i] <= 1e-12:
            break
        new[i] = 1; x[i] = 1; b -= c[i]; n[part[i]] -= 1
        cur = np.minimum(1.0, cur + A[:, i])
    return new


def static_key_greedy(key, c, budget, part, cap, avail):
    """Algorithm 1 in numpy (static ranking)."""
    order = np.argsort(-key, kind="stable")
    x = np.zeros(len(c)); b = float(budget); n = cap.astype(float).copy()
    for i in order:
        if avail[i] > 0 and c[i] <= b + 1e-9 and n[part[i]] >= 1:
            x[i] = 1; b -= c[i]; n[part[i]] -= 1
    return x


def milp_multistage(A, w, c, stage_budgets, part, cap, gamma=1.0, time_limit=120.0):
    """max sum_t gamma^t f(X_t), X_1 ⊆ ... ⊆ X_T, c(X_t) <= sum_{s<=t} b_s,
    |X_t ∩ V_p| <= cap_p. Returns (value, X (T,N), status_message).
    Single-stage is T = 1."""
    D, N = A.shape
    T = len(stage_budgets)
    P = len(cap)
    nx, nz = T * N, T * D
    xi = lambda t, j: t * N + j
    zi = lambda t, d: nx + t * D + d
    obj = np.zeros(nx + nz)
    for t in range(T):
        obj[nx + t * D: nx + (t + 1) * D] = -(gamma ** t) * w
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    cum = np.cumsum(stage_budgets)
    for t in range(T):
        for j in range(N):                                       # budget
            rows.append(r); cols.append(xi(t, j)); vals.append(c[j])
        lb.append(-np.inf); ub.append(cum[t]); r += 1
        for p in range(P):                                       # parts
            for j in np.where(part == p)[0]:
                rows.append(r); cols.append(xi(t, j)); vals.append(1.0)
            lb.append(-np.inf); ub.append(cap[p]); r += 1
        for d in range(D):                                       # coverage link
            rows.append(r); cols.append(zi(t, d)); vals.append(1.0)
            for j in np.nonzero(A[d])[0]:
                rows.append(r); cols.append(xi(t, j)); vals.append(-A[d, j])
            lb.append(-np.inf); ub.append(0.0); r += 1
        if t > 0:                                                # nestedness
            for j in range(N):
                rows += [r, r]; cols += [xi(t - 1, j), xi(t, j)]; vals += [1.0, -1.0]
                lb.append(-np.inf); ub.append(0.0); r += 1
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nx + nz))
    integrality = np.r_[np.ones(nx), np.zeros(nz)]
    bounds = Bounds(np.zeros(nx + nz), np.ones(nx + nz))
    res = milp(obj, constraints=LinearConstraint(M, lb, ub), integrality=integrality,
               bounds=bounds, options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return float("nan"), None, res.message
    X = np.round(res.x[:nx]).reshape(T, N)
    return float(-res.fun), X, res.message
