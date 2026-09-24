"""Amortised follower policy pi_phi(. | X), Assumption select + Amortisation.

Independent PPO with parameter sharing across operators (IPPO). Each operator
observes its own site, its rivals' last prices and the configuration through
its obs features, so one network covers every configuration X sampled during
training. Training on a single fixed X gives the per-configuration retrained
follower used to measure amortisation bias (experiments/exp4_amortisation.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from env.instance import CorridorInstance, random_feasible_config
from env.operations import CorridorOps, OBS_DIM, ACT_DIM


def mlp(i, o, h=64, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.Tanh()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class FollowerPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.pi = mlp(OBS_DIM, ACT_DIM, hidden)
        self.v = mlp(OBS_DIM, 1, hidden)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False):
        o = torch.as_tensor(obs)
        d = self.dist(o)
        u = d.mean if deterministic else d.sample()
        return torch.tanh(u).numpy(), u.numpy(), d.log_prob(u).sum(-1).numpy(), self.v(o).squeeze(-1).numpy()


def run_episode(env: CorridorOps, policy: FollowerPolicy, x, deterministic=False,
                demand_scale=None, override=None):
    """One stage episode. override(agent_idx, obs, act) -> act lets NashConv
    replace one agent's action. Returns trajectory dict and env metrics."""
    obs = env.reset(x, demand_scale)
    traj = {k: [] for k in ("obs", "u", "logp", "val", "rew")}
    if env.n_agents == 0:
        for _ in range(env.H):
            env.step(np.zeros((0, 3)))
        return traj, env.metrics(), env.agent_return.copy()
    done = False
    while not done:
        a, u, lp, v = policy.act(obs, deterministic)
        if override is not None:
            a = override(obs, a)
        nobs, r, done, _ = env.step(a)
        for k, val in zip(traj, (obs, u, lp, v, r)):
            traj[k].append(val)
        obs = nobs
    return traj, env.metrics(), env.agent_return.copy()


def gae(rew, val, gamma, lam):
    T = len(rew)
    adv = np.zeros_like(rew); last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def train_follower(inst: CorridorInstance, iters=150, episodes=8, fixed_config=None,
                   gamma=0.99, lam=0.95, lr=3e-4, epochs=4, minibatch=512, clip=0.2,
                   seed=0, policy=None, log_every=25, verbose=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = CorridorOps(inst, rng)
    policy = policy or FollowerPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    log = []
    for it in range(iters):
        O, U, LP, ADV, RET = [], [], [], [], []
        ep_ret = []
        for _ in range(episodes):
            if fixed_config is not None:
                x = fixed_config
            else:
                x = random_feasible_config(inst, int(rng.integers(inst.T)), rng)
            traj, met, ret = run_episode(env, policy, x)
            if not traj["obs"]:
                continue
            obs = np.stack(traj["obs"]); u = np.stack(traj["u"])
            lp = np.stack(traj["logp"]); v = np.stack(traj["val"]); r = np.stack(traj["rew"])
            for i in range(obs.shape[1]):
                adv, ret_i = gae(r[:, i], v[:, i], gamma, lam)
                O.append(obs[:, i]); U.append(u[:, i]); LP.append(lp[:, i])
                ADV.append(adv); RET.append(ret_i)
            ep_ret.append(ret.mean())
        if not O:
            continue
        O, U, LP, ADV, RET = (torch.as_tensor(np.concatenate(z), dtype=torch.float32)
                              for z in (O, U, LP, ADV, RET))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(O)
        for _ in range(epochs):
            for idx in torch.randperm(n).split(minibatch):
                d = policy.dist(O[idx])
                lp_new = d.log_prob(U[idx]).sum(-1)
                ratio = (lp_new - LP[idx]).exp()
                pl = -torch.min(ratio * ADV[idx], ratio.clamp(1 - clip, 1 + clip) * ADV[idx]).mean()
                vl = (policy.v(O[idx]).squeeze(-1) - RET[idx]).pow(2).mean()
                loss = pl + 0.5 * vl - 1e-3 * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
        log.append(float(np.mean(ep_ret)))
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"[follower] it {it:4d}  mean agent return {log[-1]:8.3f}")
    return policy, log
