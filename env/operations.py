"""Follower level: general-sum game among station operators within one stage.

Agents = sites with at least one serving module (MCS or BSS). Per interval each
agent sets  a = (price, BESS dispatch u in [-1,1], BSS charge rate iota in [0,1]),
all encoded in [-1,1]^3. Freight flow responds through the UE of env/ue.py, so
one operator's price moves queues, grid draw and swap inventory at its rivals.

Operator reward ($/100): revenue - energy bill - utility excess-draw charge - SoH cost.
Leader-level costs: C_grid = sum excess^2 (MW^2 h), C_delay = late truck-hours,
unserved = trucks taking the outside option.
V2G feed-in and PV are deferred (scoped out, see paper Section 1 of the review).
"""
from __future__ import annotations

import numpy as np

from .instance import CorridorInstance
from .ue import solve_ue

OBS_DIM = 14
ACT_DIM = 3


def demand_profile(H):
    tau = np.arange(H)
    return 0.55 + 0.45 * np.clip(np.sin(np.pi * (tau - 4) / 16), 0, None)


def tou_profile(H):
    tau = np.arange(H) % 24
    return np.where((tau >= 17) & (tau < 21), 0.25, np.where((tau >= 7) & (tau < 17), 0.14, 0.08))


class CorridorOps:
    def __init__(self, inst: CorridorInstance, rng=None):
        self.inst, self.p = inst, inst.phys
        self.H = int(self.p["H"])
        self.rng = rng or np.random.default_rng(0)
        self.dprof, self.tou = demand_profile(self.H), tou_profile(self.H)

    # ------------------------------------------------------------------
    def reset(self, x_built: np.ndarray, demand_scale: float | None = None):
        inst, p = self.inst, self.p
        cnt = inst.counts(x_built)
        self.n_mcs = cnt[:, 0] + cnt[:, 1]
        self.n_bss, self.n_bess = cnt[:, 2], cnt[:, 3]
        self.active = np.where(self.n_mcs + self.n_bss > 0)[0]
        self.n_agents = len(self.active)
        self.inv_total = self.n_bss * p["bss_inventory"]
        self.charged = self.inv_total.copy().astype(float)
        self.soc = 0.5 * self.n_bess * p["bess_energy"]
        self.last_price = np.full(inst.m, 0.5 * (p["price_lo"] + p["price_hi"]))
        self.last_util = np.zeros(inst.m)
        self.tau = 0
        self.scale = demand_scale if demand_scale is not None else float(self.rng.lognormal(0.0, 0.25))
        self.tot = dict(grid=0.0, delay=0.0, unserved=0.0, demand=0.0, served=0.0,
                        profit=0.0, soh=0.0, peak=0.0)
        self.agent_return = np.zeros(self.n_agents)
        return self._obs()

    def _obs(self):
        inst, p, a = self.inst, self.p, self.active
        if self.n_agents == 0:
            return np.zeros((0, OBS_DIM), np.float32)
        ph = 2 * np.pi * self.tau / self.H
        rp = np.array([self.last_price[inst.rivals[j]].mean() if inst.rivals[j].any()
                       else self.last_price[j] for j in a])
        nrm = lambda pr: (pr - p["price_lo"]) / (p["price_hi"] - p["price_lo"])
        o = np.stack([
            np.full(len(a), np.sin(ph)), np.full(len(a), np.cos(ph)),
            inst.site_pos[a] / inst.L, self.n_mcs[a] / 2, self.n_bss[a] / 2, self.n_bess[a] / 2,
            self.soc[a] / np.maximum(self.n_bess[a] * p["bess_energy"], 1e-9),
            self.charged[a] / np.maximum(self.inv_total[a], 1e-9),
            nrm(self.last_price[a]), nrm(rp), np.minimum(self.last_util[a], 3.0),
            np.full(len(a), self.tou[self.tau % self.H] / 0.25), inst.grid_limit[a] / 2,
            inst.rivals[a].sum(1) / 5.0], 1)
        return o.astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, act: np.ndarray):
        inst, p, a = self.inst, self.p, self.active
        m, E = inst.m, p["E_truck"]
        act = np.clip(np.asarray(act, dtype=float).reshape(self.n_agents, 3), -1, 1)
        price = np.full(m, p["price_hi"]); u = np.zeros(m); iota = np.zeros(m)
        price[a] = p["price_lo"] + 0.5 * (act[:, 0] + 1) * (p["price_hi"] - p["price_lo"])
        u[a] = act[:, 1]; iota[a] = 0.5 * (act[:, 2] + 1)

        cap_mcs = self.n_mcs * p["mcs_power"] / E
        cap_bss = np.minimum(self.n_bss * p["bss_rate"], self.charged)
        cap = cap_mcs + cap_bss
        act_site = cap > 1e-6
        share_m = np.where(act_site, cap_mcs / np.maximum(cap, 1e-9), 0.0)
        t0 = share_m * p["t0_mcs"] + (1 - share_m) * p["t0_bss"]
        feas = inst.od_sites & act_site[None, :]
        demand = inst.od_flow * self.dprof[self.tau] * self.scale
        h, v, W, gap = solve_ue(demand, feas, p["vot"] * inst.detour, price * 1000 * E,
                                np.where(act_site, t0, 1.0), cap, p["vot"], p["out_cost"],
                                p["bpr_a"], p["bpr_b"], iters=p.get("ue_iters", 80))
        v_m, v_b = v * share_m, v * (1 - share_m)
        served_m = np.minimum(v_m, cap_mcs)
        P_mcs = served_m * E
        swaps = np.minimum(v_b, self.charged)
        self.charged -= swaps
        depleted = self.inv_total - self.charged
        can = iota * self.n_bss * p["bss_charger"] / E
        bc = np.minimum(depleted, can)
        self.charged += bc
        P_bss = bc * E
        load = P_mcs + P_bss
        dis = np.minimum(np.minimum(np.maximum(u, 0) * self.n_bess * p["bess_power"], self.soc), load)
        ch = np.minimum(np.maximum(-u, 0) * self.n_bess * p["bess_power"],
                        self.n_bess * p["bess_energy"] - self.soc)
        self.soc += ch - dis
        P_grid = load + ch - dis
        excess = np.maximum(0.0, P_grid - inst.grid_limit)
        C_grid = excess ** 2

        # SoH (Wang et al. 2011 form, first-order increment in throughput)
        c_rate = np.where(self.n_bss > 0, iota * p["bss_charger"] / E, 0.0)
        T_b = p["wang_T"] + 5.0 * c_rate
        fac = np.exp((-p["wang_Ea"] + p["wang_eta"] * c_rate) / (p["wang_R"] * T_b)) / \
            np.exp((-p["wang_Ea"] + p["wang_eta"] * 1.0) / (p["wang_R"] * (p["wang_T"] + 5.0)))
        C_soh = p["soh_cost_per_mwh"] * fac * P_bss

        revenue = price * 1000 * E * (served_m + swaps)
        bill = self.tou[self.tau] * 1000 * P_grid
        profit = revenue - bill - p["kappa_grid"] * C_grid - C_soh
        r = profit[a] / 100.0

        late = np.maximum(0.0, W[None, :] + inst.detour - inst.od_slack[:, None])
        C_delay = float((h[:, :-1] * np.where(feas, late, 0.0)).sum())

        self.tot["grid"] += float(C_grid.sum()); self.tot["delay"] += C_delay
        overflow = float((v_m - served_m).sum() + (v_b - swaps).sum())
        self.tot["unserved"] += float(h[:, -1].sum()) + overflow; self.tot["demand"] += float(demand.sum())
        self.tot["served"] += float((served_m + swaps).sum()); self.tot["profit"] += float(profit.sum())
        self.tot["soh"] += float(C_soh.sum()); self.tot["peak"] = max(self.tot["peak"], float(P_grid.max(initial=0)))
        self.agent_return += r
        self.last_price = price
        self.last_util = np.where(act_site, v / np.maximum(cap, 1e-9), 0.0)
        self.tau += 1
        done = self.tau >= self.H
        return self._obs(), r.astype(np.float32), done, {"ue_gap": gap}

    def metrics(self):
        t = dict(self.tot)
        t["served_frac"] = 1.0 - t["unserved"] / max(t["demand"], 1e-9)
        return t
