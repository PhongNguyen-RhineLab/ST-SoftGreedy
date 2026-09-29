"""Constrained leader training, eq. (cmdp), identical for every action layer.

Actor  : per-module scorer y_j = g_Phi(feat_j, glob)  (permutation equivariant).
Critic : Q_h(S, x) for heads h in {r, grid, delay, tail, viol}, deep-sets over
         (feat_j, x_j, glob), regressed on Monte-Carlo returns (T is small).
Actor update
  pathwise layers (stsg*, qp_round, gumbel_topk, perturbed, penalty, lagrangian):
      maximise  L(S, x(y)) = Q_r - sum_h mu_h Q_h   through the layer's x
  score-function layer (rand_greedy):
      maximise  (L(S, x_sample) - L(S, x_greedy)) * log p(x_sample)
Duals: mu_h <- [mu_h + lr (J_h - beta_h)]_+ ; CVaR by Rockafellar-Uryasev with
eta = empirical alpha-quantile of episode grid cost, tail = (C - eta)_+.
penalty: fixed weight kappa on Q_viol. lagrangian: dual on Q_viol with beta = 0.
All costs are divided by reference values from a calibration run so that
beta factors are dimensionless.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from stsg.baselines import make_layer
from stsg.coverage import coverage_np
from env.leader_env import LeaderEnv, stack_states, FEAT_DIM, GLOB_DIM

HEADS = ["r", "grid", "delay", "tail", "viol"]


def mlp(i, o, h=128, n=2, act=nn.ReLU):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), act()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class ScoreNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.f = mlp(FEAT_DIM + GLOB_DIM, 1, h)

    def forward(self, feats, glob):
        g = glob.unsqueeze(1).expand(-1, feats.shape[1], -1)
        return self.f(torch.cat([feats, g], -1)).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.phi = mlp(FEAT_DIM + 1 + GLOB_DIM, h, h)
        self.rho = mlp(2 * h + GLOB_DIM, len(HEADS), h)

    def forward(self, feats, x, glob):
        g = glob.unsqueeze(1).expand(-1, feats.shape[1], -1)
        e = self.phi(torch.cat([feats, x.unsqueeze(-1), g], -1))
        z = torch.cat([e.mean(1), (e * x.unsqueeze(-1)).sum(1) / feats.shape[1], glob], -1)
        return self.rho(z)                                   # (B, len(HEADS))


@dataclass
class LeaderConfig:
    layer: str = "stsg"
    layer_kw: dict = field(default_factory=dict)
    iters: int = 150
    episodes_per_iter: int = 16
    gamma: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    critic_epochs: int = 20
    buffer: int = 4000
    explore_sigma: float = 0.3
    explore_decay: float = 0.99
    alpha_cvar: float = 0.9
    # constraint levels as multiples of the calibration plan (coverage greedy):
    #   expected discounted grid excess  <= 0.8 x reference       (20% reduction, binds for the reference)
    #   expected discounted mean lateness <= 1.5 x reference
    #   CVaR_0.9 of episode grid excess   <= 1.2 x reference MEAN episode grid excess
    # Fixed with experiments/exp3_bilevel.py --probe before training any learner; see README.
    beta_grid: float = 0.8
    beta_delay: float = 1.5
    beta_cvar: float = 1.2
    lr_dual: float = 0.05
    use_cvar: bool = True
    penalty_kappa: float = 5.0
    seed: int = 0


class LeaderAgent:
    def __init__(self, env: LeaderEnv, cfg: LeaderConfig, ref: dict):
        self.env, self.cfg, self.ref = env, cfg, ref
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.actor, self.critic = ScoreNet(), Critic()
        self.layer = make_layer(cfg.layer, **cfg.layer_kw)
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.oc = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.mu = {"grid": 0.0, "delay": 0.0, "tail": 0.0, "viol": 0.0}
        if cfg.layer == "penalty":
            self.mu["viol"] = cfg.penalty_kappa
        self.sigma = cfg.explore_sigma
        self.buf = []
        self.eta = 0.0

    # ------------------------------------------------------------------
    def act(self, state, explore):
        feats, glob, cb = stack_states([state])
        with torch.no_grad():
            y = self.actor(feats, glob)
            if explore and not getattr(self.layer, "stochastic", False):
                y = y + self.sigma * torch.randn_like(y)
            out = self.layer(y, cb, explore=explore)
        return out.deploy[0].numpy()

    def rollout(self, explore=True):
        s = self.env.reset(); steps = []
        done = False
        while not done:
            x = self.act(s, explore)
            s2, r, c, done, info = self.env.step(x)
            steps.append({"state": s, "x": x, "r": r, "c": c, "info": info})
            s = s2
        return steps

    # ------------------------------------------------------------------
    def _targets(self, episodes):
        g, a = self.cfg.gamma, self.cfg.alpha_cvar
        C_ep = np.array([sum(st["c"]["grid"] for st in ep) / self.ref["grid"] for ep in episodes])
        self.eta = float(np.quantile(C_ep, a))
        rows = []
        for ep, Ce in zip(episodes, C_ep):
            T = len(ep)
            tail = max(0.0, Ce - self.eta)
            ret = {h: 0.0 for h in HEADS}
            for t in reversed(range(T)):
                st = ep[t]
                inc = {"r": st["r"], "grid": st["c"]["grid"] / self.ref["grid_disc"],
                       "delay": st["c"]["delay"] / self.ref["delay_disc"], "tail": 0.0,
                       "viol": st["c"]["viol"]}
                for h in HEADS:
                    ret[h] = inc[h] + g * ret[h]
                ret["tail"] = tail
                rows.append((st["state"], st["x"], [ret[h] for h in HEADS]))
        return rows, C_ep

    def _lagrangian(self, q):
        mu = self.mu
        L = q[:, 0] - mu["grid"] * q[:, 1] - mu["delay"] * q[:, 2] - mu["viol"] * q[:, 4]
        if self.cfg.use_cvar:
            L = L - mu["tail"] / (1 - self.cfg.alpha_cvar) * q[:, 3]
        return L

    def update(self, episodes):
        cfg = self.cfg
        rows, C_ep = self._targets(episodes)
        self.buf = (self.buf + rows)[-cfg.buffer:]
        # critic
        feats, glob, cb = stack_states([r[0] for r in self.buf])
        X = torch.as_tensor(np.stack([r[1] for r in self.buf]), dtype=torch.float32)
        Y = torch.as_tensor(np.array([r[2] for r in self.buf]), dtype=torch.float32)
        for _ in range(cfg.critic_epochs):
            idx = torch.randperm(len(X))[:512]
            loss = (self.critic(feats[idx], X[idx], glob[idx]) - Y[idx]).pow(2).mean()
            self.oc.zero_grad(); loss.backward(); self.oc.step()
        # actor on fresh states
        feats, glob, cb = stack_states([r[0] for r in rows])
        y = self.actor(feats, glob)
        if cfg.layer == "rand_greedy":
            out = self.layer(y, cb, explore=True)
            with torch.no_grad():
                greedy = self.layer(y.detach(), cb, explore=False).deploy
                adv = self._lagrangian(self.critic(feats, out.x, glob)) - \
                    self._lagrangian(self.critic(feats, greedy, glob))
            a_loss = -(adv * out.logp).mean()
        else:
            out = self.layer(y, cb, explore=getattr(self.layer, "stochastic", False))
            a_loss = -self._lagrangian(self.critic(feats, out.x, glob)).mean()
        self.oa.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.oa.step()
        # duals
        g = cfg.gamma
        J = {h: np.mean([sum(g ** t * st["c"][h] for t, st in enumerate(ep)) for ep in episodes])
             for h in ("grid", "delay", "viol")}
        Jg, Jd = J["grid"] / self.ref["grid_disc"], J["delay"] / self.ref["delay_disc"]
        cvar = self.eta + np.mean(np.maximum(0, C_ep - self.eta)) / (1 - cfg.alpha_cvar)
        self.mu["grid"] = max(0.0, self.mu["grid"] + cfg.lr_dual * (Jg - cfg.beta_grid))
        self.mu["delay"] = max(0.0, self.mu["delay"] + cfg.lr_dual * (Jd - cfg.beta_delay))
        if cfg.use_cvar:
            self.mu["tail"] = max(0.0, self.mu["tail"] + cfg.lr_dual * (cvar - cfg.beta_cvar))
        if cfg.layer == "lagrangian":
            self.mu["viol"] = max(0.0, self.mu["viol"] + cfg.lr_dual * J["viol"])
        self.sigma *= cfg.explore_decay
        return {"a_loss": a_loss.item(), "Jg": Jg, "Jd": Jd, "cvar": float(cvar), **{f"mu_{k}": v for k, v in self.mu.items()}}

    def train(self, verbose=True, log_every=10):
        log = []
        t0 = time.time()
        for it in range(self.cfg.iters):
            eps = [self.rollout(True) for _ in range(self.cfg.episodes_per_iter)]
            info = self.update(eps)
            info["ret"] = float(np.mean([sum(self.cfg.gamma ** t * s["r"] for t, s in enumerate(e)) for e in eps]))
            info["infeas"] = float(np.mean([s["c"]["infeasible"] for e in eps for s in e]))
            info["time"] = time.time() - t0
            log.append(info)
            if verbose and (it % log_every == 0 or it == self.cfg.iters - 1):
                print(f"[leader:{self.cfg.layer}] it {it:4d} ret {info['ret']:.3f} infeas {info['infeas']:.2f} "
                      f"Jg {info['Jg']:.2f} Jd {info['Jd']:.2f} cvar {info['cvar']:.2f} "
                      f"mu {info['mu_grid']:.2f}/{info['mu_delay']:.2f}/{info['mu_tail']:.2f}/{info['mu_viol']:.2f}")
        return log

    # ------------------------------------------------------------------
    def evaluate(self, n=None):
        """Deterministic policy over every demand quantile (common random numbers)."""
        return evaluate_policy(self.env, lambda st, env: self.act(st, explore=False),
                               self.cfg.gamma, self.cfg.alpha_cvar, ref=self.ref, cfg=self.cfg)


# ----------------------------------------------------------------------
# evaluation shared by learned leaders and fixed reference plans
# ----------------------------------------------------------------------
def evaluate_policy(env: LeaderEnv, act_fn, gamma=0.95, alpha_cvar=0.9, ref=None, cfg=None):
    """act_fn(state, env) -> binary module vector for the current stage.
    Runs one episode per demand quantile k; identical protocol for every row
    of the main table, so references and learned layers are comparable.

    With ref and cfg, also reports the CMDP constraints exactly as the dual
    ascent in LeaderAgent.update defines them:
      g_grid  = E[sum_t gamma^t C_grid_t]  / ref_grid   vs beta_grid
      g_delay = E[sum_t gamma^t C_delay_t] / ref_delay  vs beta_delay
      g_cvar  = CVaR_alpha[sum_t C_grid_t] / ref_grid   vs beta_cvar (if use_cvar)
    ref_grid_disc / ref_delay_disc are the DISCOUNTED expected costs of the
    calibration plan and ref_grid its undiscounted episode total, so the
    calibration plan itself has g/beta = 1/beta on the two expectation constraints.
    Expectation = mean over the K demand quantiles, which is the distribution
    the training episodes sample from."""
    res = []
    for k in range(len(env.scales)):
        s = env.reset(); env.k = k
        done = False; t = 0
        acc = {"ret": 0.0, "grid": 0.0, "delay": 0.0, "viol": 0.0, "infeas": 0, "cov_disc": 0.0,
               "grid_disc": 0.0, "delay_disc": 0.0}
        while not done:
            x = act_fn(s, env)
            s, r, c, done, info = env.step(x)
            acc["ret"] += gamma ** t * r; acc["grid"] += c["grid"]; acc["delay"] += c["delay"]
            acc["grid_disc"] += gamma ** t * c["grid"]; acc["delay_disc"] += gamma ** t * c["delay"]
            acc["viol"] += c["viol"]; acc["infeas"] += c["infeasible"]
            acc["cov_disc"] += gamma ** t * info["coverage"]
            t += 1
        acc["served_frac"] = info["served_frac"]; acc["cov_final"] = info["coverage"]
        acc["built"] = info["built"].tolist()
        res.append(acc)
    grid = np.array([r["grid"] for r in res])
    q = np.quantile(grid, alpha_cvar)
    out = {k: float(np.mean([r[k] for r in res])) for k in
           ("ret", "grid", "delay", "viol", "infeas", "cov_disc", "cov_final", "served_frac",
            "grid_disc", "delay_disc")}
    out["infeas_rate"] = out.pop("infeas") / env.inst.T
    out["cvar_grid"] = float(q + np.mean(np.maximum(0, grid - q)) / (1 - alpha_cvar))
    out["built_example"] = res[0]["built"]
    if ref is not None and cfg is not None:
        out.update(constraint_report(out, ref, cfg))
    return out


def constraint_report(ev, ref, cfg, tol=1e-6):
    """Ratios g/beta (<= 1 means satisfied) and the joint CMDP-feasibility flag."""
    r = {"g_grid": ev["grid_disc"] / ref["grid_disc"] / cfg.beta_grid,
         "g_delay": ev["delay_disc"] / ref["delay_disc"] / cfg.beta_delay,
         "g_cvar": ev["cvar_grid"] / ref["grid"] / cfg.beta_cvar}
    active = ["g_grid", "g_delay"] + (["g_cvar"] if cfg.use_cvar else [])
    r["cmdp_ok"] = float(all(r[k] <= 1 + tol for k in active) and ev["infeas_rate"] == 0)
    r["cmdp_max_ratio"] = max(r[k] for k in active)
    return r


def reference_policies():
    """Fixed, non-learned plans evaluated with evaluate_policy.

    greedy_rerank : per stage, re-ranking coverage greedy (best of gain and
                    density) on the residual budget. Optimises coverage only,
                    blind to grid / delay / the follower game. Same plan that
                    calibrate() uses to set the cost references, so its
                    normalised grid and delay costs are 1 by construction.
    static_key    : Algorithm 1 with the modular coverage key and no learning,
                    i.e. what STSG deploys if the scorer outputs that key.
                    Measures what the leader learned beyond the hand-made key.
    """
    from stsg.coverage import rerank_greedy, static_key_greedy

    def greedy_rerank(state, env):
        inst = env.inst
        budget, cap, avail = inst.residual(env.built, env.t)
        return rerank_greedy(inst.A, inst.w, inst.cost, budget, inst.mod_site, cap, avail,
                             base=env.built, rule="best")

    def static_key(state, env):
        inst = env.inst
        budget, cap, avail = inst.residual(env.built, env.t)
        key = (inst.w[:, None] * inst.A).sum(0)
        return static_key_greedy(key, inst.cost, budget, inst.mod_site, cap, avail)

    return {"ref_greedy_rerank": greedy_rerank, "ref_static_key": static_key}


def budget_scaled_greedy(phi):
    """Re-ranking coverage greedy restricted to a fraction phi of the cumulative
    budget at every stage. phi < 1 builds less, hence lower grid / delay cost."""
    from stsg.coverage import rerank_greedy

    def act(state, env):
        inst = env.inst
        budget, cap, avail = inst.residual(env.built, env.t)
        cum = float(inst.stage_budget[: env.t + 1].sum())
        spent = cum - budget
        b_phi = max(0.0, phi * cum - spent)
        return rerank_greedy(inst.A, inst.w, inst.cost, b_phi, inst.mod_site, cap, avail,
                             base=env.built, rule="best")
    return act


def budget_scaled_greedy_bess(phi):
    """Domain heuristic: budget-scaled coverage greedy, then the remaining stage
    budget buys BESS (grid-peak shaving) at the sites with the most MCS blocks."""
    from stsg.coverage import rerank_greedy

    def act(state, env):
        inst = env.inst
        budget, cap, avail = inst.residual(env.built, env.t)
        cum = float(inst.stage_budget[: env.t + 1].sum())
        new = rerank_greedy(inst.A, inst.w, inst.cost, max(0.0, phi * cum - (cum - budget)),
                            inst.mod_site, cap, avail, base=env.built, rule="best")
        tot = np.clip(env.built + new, 0, 1)
        b = budget - float((inst.cost * new).sum())
        n = cap - np.bincount(inst.mod_site, weights=new, minlength=inst.m)
        cnt = inst.counts(tot)
        mcs = cnt[:, 0] + cnt[:, 1]
        for j in np.argsort(-mcs, kind="stable"):
            if mcs[j] == 0:
                break
            i = int(np.where((inst.mod_site == j) & (inst.mod_type == 3))[0][0])
            if tot[i] == 0 and inst.cost[i] <= b + 1e-9 and n[j] >= 1:
                new[i] = 1.0; b -= inst.cost[i]; n[j] -= 1
        return new
    return act


def constrained_greedy_reference(env, ref, cfg, phis=None):
    """ref_greedy_constrained: sweep phi over two heuristic families (coverage
    greedy, coverage greedy + BESS), keep the best-return plan that satisfies
    the CMDP constraints. The sweep is scored on the evaluation quantiles
    themselves, i.e. tuned on the test distribution: this favours the reference,
    which is the conservative direction for any claim that a learned leader beats it.
    Returns (best_eval_with_phi, sweep list). If no phi is CMDP-feasible the
    least-violating plan is returned with cmdp_ok = 0."""
    phis = phis if phis is not None else [round(0.1 * i, 1) for i in range(0, 11)]
    sweep = []
    for fam, maker in (("greedy", budget_scaled_greedy), ("greedy+BESS", budget_scaled_greedy_bess)):
        for phi in phis:
            ev = evaluate_policy(env, maker(phi), cfg.gamma, cfg.alpha_cvar, ref=ref, cfg=cfg)
            ev["phi"], ev["family"] = phi, fam
            sweep.append(ev)
    ok = [e for e in sweep if e["cmdp_ok"]]
    best = max(ok, key=lambda e: e["ret"]) if ok else min(sweep, key=lambda e: e["cmdp_max_ratio"])
    keys = ("family", "phi", "ret", "g_grid", "g_delay", "g_cvar", "cmdp_ok")
    return best, [{k: e[k] for k in keys} for e in sweep]


def calibrate(env: LeaderEnv, gamma=0.95):
    """Reference costs of the re-ranking coverage greedy run stage by stage.

    grid, delay           : mean undiscounted episode totals (CVaR is on the undiscounted total)
    grid_disc, delay_disc : mean discounted totals, same discounting as the constraint
    The floors only keep the ratios finite if the reference plan is (nearly) cost-free."""
    from stsg.coverage import rerank_greedy
    inst = env.inst
    acc = {"grid": [], "delay": [], "grid_disc": [], "delay_disc": []}
    for k in range(len(env.scales)):
        env.reset(); env.k = k; done = False; t = 0
        tot = dict.fromkeys(acc, 0.0)
        while not done:
            budget, cap, avail = inst.residual(env.built, env.t)
            new = rerank_greedy(inst.A, inst.w, inst.cost, budget, inst.mod_site, cap, avail,
                                base=env.built, rule="best")
            _, _, c, done, _ = env.step(new)
            tot["grid"] += c["grid"]; tot["delay"] += c["delay"]
            tot["grid_disc"] += gamma ** t * c["grid"]; tot["delay_disc"] += gamma ** t * c["delay"]
            t += 1
        for key in acc:
            acc[key].append(tot[key])
    floor = {"grid": 1.0, "grid_disc": 1.0, "delay": 1e-3, "delay_disc": 1e-3}
    return {key: max(float(np.mean(v)), floor[key]) for key, v in acc.items()}
