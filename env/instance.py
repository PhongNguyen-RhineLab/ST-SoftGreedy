"""Parametric corridor generator (Experimental Protocol, Instances).

m candidate sites on a line corridor, 4 modules per site (N = 4m):
  type 0 MCS-A  first megawatt charger block
  type 1 MCS-B  second megawatt charger block
  type 2 BSS    battery-swap bay with its own battery inventory
  type 3 BESS   stationary storage, no truck service, shaves grid peaks
Parts of the partition matroid are sites (grid-connection slots cap_p).
Costs are affinely rescaled so that c_max / c_min = theta exactly.

All demand is synthetic. The paper text mentions a published freight OD matrix;
no such matrix is loaded here, so either add a loader or change that sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from stsg.layer import ConstraintBatch

N_TYPES = 4
TYPE_NAMES = ["MCS-A", "MCS-B", "BSS", "BESS"]
BASE_COST = np.array([1.0, 0.9, 1.6, 0.7])
COVER_W = np.array([0.6, 0.4, 0.7, 0.0])

DEFAULT_PHYS = dict(
    H=24,                       # operating intervals (hours) per stage episode
    E_truck=0.5,                # MWh per stop
    mcs_power=1.0,              # MW per MCS block
    t0_mcs=0.5, t0_bss=0.15,    # service time (h)
    bss_rate=6.0,               # swaps per hour per bay
    bss_inventory=6,            # batteries per bay
    bss_charger=1.0,            # MW charging power per bay
    bess_energy=2.0, bess_power=1.0,
    price_lo=0.15, price_hi=0.60,     # $/kWh
    vot=80.0,                   # $/truck-hour
    out_cost=450.0,             # $ outside option (unserved)
    bpr_a=0.15, bpr_b=4.0,
    kappa_grid=200.0,           # $ per MW^2 h of excess draw (utility charge)
    soh_cost_per_mwh=8.0,       # $ per MWh throughput at 1C, 25C (scale)
    # Wang et al. 2011 semi-empirical law constants -- VERIFY against the paper
    wang_Ea=31700.0, wang_eta=370.3, wang_R=8.314, wang_T=298.15, wang_z=0.55,
)


@dataclass
class CorridorInstance:
    m: int
    T: int
    theta: float
    L: float
    site_pos: np.ndarray
    mod_site: np.ndarray
    mod_type: np.ndarray
    cost: np.ndarray
    cap: np.ndarray
    stage_budget: np.ndarray
    od_o: np.ndarray
    od_d: np.ndarray
    od_flow: np.ndarray
    od_slack: np.ndarray
    od_sites: np.ndarray          # (O,m) bool feasible stop sites
    detour: np.ndarray            # (O,m) hours
    A: np.ndarray                 # (O,N) coverage coefficients
    w: np.ndarray                 # (O,)
    grid_limit: np.ndarray        # (m,) MW
    rivals: np.ndarray            # (m,m) bool: share at least one OD
    phys: dict = field(default_factory=lambda: dict(DEFAULT_PHYS))

    @property
    def N(self):
        return len(self.cost)

    def counts(self, x: np.ndarray) -> np.ndarray:
        """(m,4) number of built modules of each type per site."""
        out = np.zeros((self.m, N_TYPES))
        np.add.at(out, (self.mod_site, self.mod_type), x)
        return out

    def residual(self, built: np.ndarray, t: int):
        spent = float((self.cost * built).sum())
        budget = float(self.stage_budget[: t + 1].sum()) - spent
        load = np.bincount(self.mod_site, weights=built, minlength=self.m)
        cap = self.cap - load
        avail = 1.0 - built
        return budget, cap, avail

    def constraint_batch(self, built: np.ndarray, t: int, device="cpu") -> ConstraintBatch:
        budget, cap, avail = self.residual(built, t)
        f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device).unsqueeze(0)
        return ConstraintBatch(f(self.cost), f([budget]).squeeze(0), f(self.mod_site, torch.long),
                               f(cap), f(avail))


def generate(m=8, theta=5.0, T=3, seed=0, L=None, od_per_site=3, truck_range=420.0,
             cap_per_site=2, budget_frac=0.2, window_lo=0.5, phys=None) -> CorridorInstance:
    rng = np.random.default_rng(seed)
    L = float(L) if L is not None else 100.0 * m + 1.9 * truck_range   # ~constant site density
    site_pos = np.sort(rng.uniform(20, L - 20, size=m))
    mod_site = np.repeat(np.arange(m), N_TYPES)
    mod_type = np.tile(np.arange(N_TYPES), m)
    raw = BASE_COST[mod_type] * rng.uniform(0.8, 1.25, size=m * N_TYPES)
    cost = 1.0 + (theta - 1.0) * (raw - raw.min()) / (raw.max() - raw.min())
    total = budget_frac * cost.sum()
    stage_budget = np.full(T, max(total / T, 2.0 * cost.max() + 1e-6))  # Lemma psystem: B_t >= 2 c_max

    O = od_per_site * m
    od_o = rng.uniform(0, L - 1.9 * truck_range, size=O)      # one stop per trip
    od_d = od_o + rng.uniform(truck_range + 50, 1.9 * truck_range, size=O)
    od_flow = rng.uniform(0.3, 1.2, size=O)
    od_slack = rng.uniform(0.3, 1.5, size=O)
    lo = od_o + window_lo * truck_range
    hi = np.minimum(od_o + truck_range, od_d)
    od_sites = (site_pos[None, :] >= lo[:, None]) & (site_pos[None, :] <= hi[:, None])
    detour = rng.uniform(0.02, 0.2, size=(O, m))
    A = COVER_W[mod_type][None, :] * od_sites[:, mod_site] / (1.0 + od_flow[:, None])
    w = od_flow.copy()
    grid_limit = rng.uniform(0.6, 1.4, size=m)       # binds for 2 MCS or MCS + BSS charging
    rivals = (od_sites.T.astype(int) @ od_sites.astype(int)) > 0
    np.fill_diagonal(rivals, False)
    p = dict(DEFAULT_PHYS)
    if phys:
        p.update(phys)
    return CorridorInstance(m, T, theta, L, site_pos, mod_site, mod_type, cost,
                            np.full(m, float(cap_per_site)), stage_budget, od_o, od_d, od_flow,
                            od_slack, od_sites, detour, A, w, grid_limit, rivals, p)


def random_feasible_config(inst: CorridorInstance, t: int, rng: np.random.Generator):
    """Feasible cumulative configuration at stage t (hard greedy on random scores)."""
    from stsg.coverage import static_key_greedy
    budget = inst.stage_budget[: t + 1].sum()
    key = rng.normal(size=inst.N) + 0.5 * (inst.mod_type < 3)
    return static_key_greedy(key, inst.cost, budget, inst.mod_site, inst.cap, np.ones(inst.N))
