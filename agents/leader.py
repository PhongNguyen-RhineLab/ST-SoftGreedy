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
    beta_grid: float = 0.8           # multiples of the calibration reference
    beta_delay: float = 0.8
    beta_cvar: float = 1.0
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
                inc = {"r": st["r"], "grid": st["c"]["grid"] / self.ref["grid"],
                       "delay": st["c"]["delay"] / self.ref["delay"], "tail": 0.0,
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
        Jg, Jd = J["grid"] / self.ref["grid"], J["delay"] / self.ref["delay"]
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
        cfg, env = self.cfg, self.env
        g = cfg.gamma
        res = []
        for k in range(len(env.scales)):
            s = env.reset(); env.k = k
            done = False; t = 0
            acc = {"ret": 0.0, "grid": 0.0, "delay": 0.0, "viol": 0.0, "infeas": 0, "cov_disc": 0.0}
            while not done:
                x = self.act(s, explore=False)
                s, r, c, done, info = env.step(x)
                acc["ret"] += g ** t * r; acc["grid"] += c["grid"]; acc["delay"] += c["delay"]
                acc["viol"] += c["viol"]; acc["infeas"] += c["infeasible"]
                acc["cov_disc"] += g ** t * info["coverage"]
                t += 1
            acc["served_frac"] = info["served_frac"]; acc["cov_final"] = info["coverage"]
            acc["built"] = info["built"].tolist()
            res.append(acc)
        grid = np.array([r["grid"] for r in res])
        q = np.quantile(grid, cfg.alpha_cvar)
        out = {k: float(np.mean([r[k] for r in res])) for k in
               ("ret", "grid", "delay", "viol", "infeas", "cov_disc", "cov_final", "served_frac")}
        out["infeas_rate"] = out.pop("infeas") / env.inst.T
        out["cvar_grid"] = float(q + np.mean(np.maximum(0, grid - q)) / (1 - cfg.alpha_cvar))
        out["built_example"] = res[0]["built"]
        return out


def calibrate(env: LeaderEnv, gamma=0.95):
    """Reference costs from the re-ranking coverage greedy run stage by stage."""
    from stsg.coverage import rerank_greedy
    inst = env.inst
    grids, delays = [], []
    for k in range(len(env.scales)):
        env.reset(); env.k = k; done = False; G = D = 0.0
        while not done:
            budget, cap, avail = inst.residual(env.built, env.t)
            new = rerank_greedy(inst.A, inst.w, inst.cost, budget, inst.mod_site, cap, avail,
                                base=env.built, rule="best")
            _, _, c, done, _ = env.step(new)
            G += c["grid"]; D += c["delay"]
        grids.append(G); delays.append(D)
    # floors keep the normalisation finite when the reference plan is (nearly) cost-free
    return {"grid": max(float(np.mean(grids)), 1.0), "delay": max(float(np.mean(delays)), 1.0)}
