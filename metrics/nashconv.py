"""NashConv of the follower profile (Assumption select).

True NashConv needs an exact best response per operator in a stochastic game,
which is intractable here. We report a restricted-deviation estimate:
  NashConv_hat = sum_i max(0, max_{d in D} R_i(d, pi_-i) - R_i(pi)),
with D = constant-price deviations on a grid (dispatch and charging kept from
pi_i), evaluated on common random numbers. Because D is a subset of all
deviations this is a LOWER bound on true NashConv, and the paper must say so.
Optionally D can include a short PPO best response (br_iters > 0).
"""
from __future__ import annotations

import numpy as np

from agents.follower import run_episode
from env.operations import CorridorOps


def nashconv(inst, policy, x, n_prices=9, n_eval=4, seed=0):
    rng = np.random.default_rng(seed)
    scales = rng.lognormal(0.0, 0.25, size=n_eval)
    env = CorridorOps(inst, np.random.default_rng(seed))
    base = np.mean([run_episode(env, policy, x, True, s)[2] for s in scales], 0)
    if len(base) == 0:
        return {"nashconv": 0.0, "rel": 0.0, "per_agent": []}
    gains = np.zeros(len(base))
    for i in range(len(base)):
        best = base[i]
        for pr in np.linspace(-1, 1, n_prices):
            def ov(obs, a, i=i, pr=pr):
                a = a.copy(); a[i, 0] = pr; return a
            dev = np.mean([run_episode(env, policy, x, True, s, override=ov)[2][i] for s in scales])
            best = max(best, dev)
        gains[i] = best - base[i]
    nc = float(gains.sum())
    return {"nashconv": nc, "rel": nc / max(np.abs(base).sum(), 1e-9), "per_agent": gains.tolist()}
