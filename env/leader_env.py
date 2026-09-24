"""Leader (macro) MDP of the C-BMDP.

State: built modules, residual budget and part capacities, stage index.
Action: binary module vector from an action layer (may be infeasible for
baselines; the environment executes it and records the violation, and any
overspend is carried into the next stage's residual budget).
Reward: R_macro_t = served_frac_t + cov_weight * f(X_t) / sum(w).
Costs:  grid (MW^2 h), delay (late truck-hours), viol (constraint excess).

Follower outcomes use the frozen amortised policy with deterministic actions.
Demand stochasticity comes from K fixed lognormal quantiles (common random
numbers), which also makes (config, k) results cacheable.
"""
from __future__ import annotations

import numpy as np
import torch

from stsg.coverage import coverage_np
from stsg.layer import ConstraintBatch
from agents.follower import run_episode
from .instance import CorridorInstance, N_TYPES
from .operations import CorridorOps

FEAT_DIM = N_TYPES + 8
GLOB_DIM = 4


class LeaderEnv:
    def __init__(self, inst: CorridorInstance, follower, cov_weight=1.0, K_scales=8,
                 seed=0, cache=True):
        self.inst, self.follower, self.cov_weight = inst, follower, cov_weight
        self.rng = np.random.default_rng(seed)
        q = (np.arange(K_scales) + 0.5) / K_scales
        from scipy.stats import norm
        self.scales = np.exp(0.25 * norm.ppf(q))
        self.ops = CorridorOps(inst, np.random.default_rng(seed))
        self.cache = {} if cache else None
        self.wsum = float(inst.w.sum())
        cs = np.zeros(inst.N)
        cs[:] = (inst.w[:, None] * inst.A).sum(0) / self.wsum
        self.cov_strength = cs
        self.rival_cnt = inst.rivals.sum(1)[inst.mod_site] / 5.0

    # ------------------------------------------------------------------
    def reset(self):
        self.built = np.zeros(self.inst.N)
        self.t = 0
        self.k = int(self.rng.integers(len(self.scales)))   # one demand draw per episode
        return self.state()

    def state(self):
        inst = self.inst
        budget, cap, avail = inst.residual(self.built, min(self.t, inst.T - 1))
        site_any = (np.bincount(inst.mod_site, weights=self.built, minlength=inst.m) > 0)
        feats = np.concatenate([
            np.eye(N_TYPES)[inst.mod_type],
            np.stack([inst.cost / inst.theta, inst.site_pos[inst.mod_site] / inst.L,
                      self.cov_strength, self.built, cap[inst.mod_site] / inst.cap[inst.mod_site],
                      site_any[inst.mod_site].astype(float), inst.grid_limit[inst.mod_site] / 2,
                      self.rival_cnt], 1)], 1).astype(np.float32)
        tot = inst.stage_budget.sum()
        glob = np.array([self.t / inst.T, budget / tot,
                         coverage_np(self.built, inst.A, inst.w) / self.wsum,
                         inst.stage_budget[min(self.t, inst.T - 1)] / tot], np.float32)
        cb = inst.constraint_batch(self.built, min(self.t, inst.T - 1))
        return {"feats": feats, "glob": glob, "cb": cb}

    def follower_outcome(self, x, k):
        key = (x.astype(np.int8).tobytes(), k)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        _, met, _ = run_episode(self.ops, self.follower, x, deterministic=True,
                                demand_scale=float(self.scales[k]))
        if self.cache is not None:
            self.cache[key] = met
        return met

    def step(self, x_deploy: np.ndarray):
        inst = self.inst
        cb = inst.constraint_batch(self.built, self.t)
        xt = torch.as_tensor(x_deploy, dtype=torch.float32).unsqueeze(0)
        viol = cb.violation(xt)
        new = x_deploy * (1 - self.built)
        self.built = np.clip(self.built + new, 0, 1)
        met = self.follower_outcome(self.built, self.k)
        cov = coverage_np(self.built, inst.A, inst.w)
        reward = met["served_frac"] + self.cov_weight * cov / self.wsum
        costs = {"grid": met["grid"], "delay": met["delay"],
                 "viol": float(viol["total"]), "infeasible": float(viol["infeasible"])}
        info = {"coverage": cov, "served_frac": met["served_frac"], "built": self.built.copy()}
        self.t += 1
        done = self.t >= inst.T
        return (None if done else self.state()), reward, costs, done, info


def stack_states(states):
    feats = torch.as_tensor(np.stack([s["feats"] for s in states]))
    glob = torch.as_tensor(np.stack([s["glob"] for s in states]))
    cbs = [s["cb"] for s in states]
    cb = ConstraintBatch(*(torch.cat([getattr(c, f) for c in cbs], 0)
                           for f in ("c", "budget", "part", "cap", "avail")))
    return feats, glob, cb
