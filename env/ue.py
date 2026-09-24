"""User equilibrium over stop choice, eq. (beckmann).

Each OD o chooses a stop s among feasible active sites, or the outside option.
Path cost  = fixed_os + price_cost_s + vot * W_s(v_s),  outside = out_cost,
W_s(v) = t0_s (1 + a (v/cap_s)^b) strictly increasing, so site flows v are unique.
Solved by Frank-Wolfe with bisection line search on the Beckmann potential.
"""
from __future__ import annotations

import numpy as np


def _wait(v, t0, cap, a, b):
    return t0 * (1.0 + a * (v / cap) ** b)


def solve_ue(demand, feasible, fixed, price_cost, t0, cap, vot, out_cost,
             a=0.15, b=4.0, iters=40, tol=1e-4):
    """demand (O,), feasible (O,S) bool, fixed (O,S), price_cost (S,), t0, cap (S,).
    Returns h (O,S+1) path flows (last column = outside), v (S,), W (S,), rel_gap."""
    O, S = feasible.shape
    capc = np.maximum(cap, 1e-6)
    base = np.where(feasible, fixed + price_cost[None, :], np.inf)

    def path_cost(v):
        W = _wait(v, t0, capc, a, b)
        pc = base + vot * W[None, :]
        return np.concatenate([pc, np.full((O, 1), out_cost)], 1), W

    def aon(pc):
        y = np.zeros((O, S + 1))
        y[np.arange(O), np.argmin(pc, 1)] = demand
        return y

    pc, W = path_cost(np.zeros(S))
    h = aon(pc)
    gap = np.inf
    for _ in range(iters):
        v = h[:, :S].sum(0)
        pc, W = path_cost(v)
        y = aon(pc)
        pcf = np.where(np.isfinite(pc), pc, 0.0)
        tstt = (h * pcf).sum()
        sptt = (y * pcf).sum()
        gap = (tstt - sptt) / max(tstt, 1e-9)
        if gap < tol:
            break
        dh = y - h
        dv = dh[:, :S].sum(0)
        const = (dh[:, :S] * np.where(feasible, base, 0.0)).sum() + dh[:, S].sum() * out_cost

        def dphi(al):
            return (dv * vot * _wait(v + al * dv, t0, capc, a, b)).sum() + const

        lo_, hi_ = 0.0, 1.0
        if dphi(1.0) <= 0:
            al = 1.0
        else:
            for _ in range(30):
                mid = 0.5 * (lo_ + hi_)
                if dphi(mid) > 0:
                    hi_ = mid
                else:
                    lo_ = mid
            al = 0.5 * (lo_ + hi_)
        h = h + al * dh
    v = h[:, :S].sum(0)
    _, W = path_cost(v)
    return h, v, W, gap
