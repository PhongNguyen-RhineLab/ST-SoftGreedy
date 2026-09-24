"""Baseline action layers (Experimental Protocol, baselines i-vi).

All layers share the interface  layer(y, cb, explore) -> LayerOut.
  x      : tensor in the training graph (binary value + surrogate gradient, or
           no gradient for score-function layers which return logp instead)
  deploy : binary action executed by the environment (may be infeasible for
           penalty / Lagrangian / QP-rounding / Gumbel top-k)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ConstraintBatch, LayerOut, hard_greedy


# (i)/(ii) independent Bernoulli with straight-through; the penalty or the
# Lagrange multiplier is applied by the trainer, not by the layer.
class BernoulliST(nn.Module):
    stochastic = True

    def forward(self, y, cb: ConstraintBatch, explore=False) -> LayerOut:
        p = torch.sigmoid(y) * cb.avail
        hard = torch.bernoulli(p.detach()) if explore else (p.detach() > 0.5).float()
        return LayerOut(x=hard + p - p.detach(), deploy=hard, aux={"p": p})


# (iv) Gumbel top-k inside every part: enforces the partition matroid, ignores
# the knapsack.
class PartTopK(nn.Module):
    stochastic = True

    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau

    def forward(self, y, cb: ConstraintBatch, explore=False) -> LayerOut:
        g = y
        if explore:
            u = torch.rand_like(y).clamp(1e-9, 1 - 1e-9)
            g = y - torch.log(-torch.log(u))
        neg = torch.finfo(y.dtype).min / 4
        g = torch.where(cb.avail > 0, g, torch.full_like(g, neg))
        same = cb.part.unsqueeze(-1) == cb.part.unsqueeze(-2)          # (B,N,N)
        idx = torch.arange(y.shape[-1], device=y.device)
        higher = (g.unsqueeze(-2) > g.unsqueeze(-1)) | (
            (g.unsqueeze(-2) == g.unsqueeze(-1)) & (idx.view(1, 1, -1) < idx.view(1, -1, 1)))
        rank = (same & higher & (cb.avail.unsqueeze(-2) > 0)).sum(-1)
        capn = cb.cap.gather(1, cb.part)
        hard = ((rank < capn) & (cb.avail > 0)).float()
        # soft: cap_p * softmax within part
        z = torch.exp((g - g.detach().amax(-1, keepdim=True)) / self.tau) * cb.avail
        denom = torch.einsum("bij,bj->bi", same.float(), z).clamp_min(1e-12)
        soft = (capn * z / denom).clamp(max=1.0)
        return LayerOut(x=hard + soft - soft.detach(), deploy=hard.detach())


# (iii) continuous projection onto the LP relaxation of X_t, then rounding.
def _proj_box(z, avail):
    return torch.minimum(z.clamp(min=0.0), avail)


def _proj_knap(z, c, budget):
    ex = F.relu((c * z).sum(-1) - budget)
    return z - (ex / (c * c).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * c


def _proj_parts(z, oh, cap):
    load = torch.einsum("bn,bnp->bp", z, oh)
    cnt = oh.sum(1).clamp_min(1.0)
    ex = F.relu(load - cap) / cnt
    return z - torch.einsum("bp,bnp->bn", ex, oh)


def dykstra_projection(z, cb: ConstraintBatch, iters=60):
    """Euclidean projection onto box ∩ knapsack ∩ partition halfspaces,
    unrolled Dykstra (differentiable by autograd)."""
    oh = cb.onehot()
    projs = [lambda v: _proj_box(v, cb.avail),
             lambda v: _proj_knap(v, cb.c, cb.budget),
             lambda v: _proj_parts(v, oh, cb.cap)]
    x = z
    inc = [torch.zeros_like(z) for _ in projs]
    for _ in range(iters):
        for j, pj in enumerate(projs):
            yj = pj(x + inc[j])
            inc[j] = x + inc[j] - yj
            x = yj
    return x


class QPProjectRound(nn.Module):
    """sigmoid(y) -> projection -> round at 1/2. Uses cvxpylayers when
    use_cvxpy=True and the package is installed, unrolled Dykstra otherwise."""
    stochastic = False

    def __init__(self, iters=60, use_cvxpy=False):
        super().__init__()
        self.iters, self.use_cvxpy = iters, use_cvxpy
        self._cvx = None

    def _cvx_layer(self, N, P):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        x = cp.Variable(N); z = cp.Parameter(N); c = cp.Parameter(N, nonneg=True)
        b = cp.Parameter(); av = cp.Parameter(N); A = cp.Parameter((P, N)); r = cp.Parameter(P)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - z)),
                          [x >= 0, x <= av, c @ x <= b, A @ x <= r])
        return CvxpyLayer(prob, parameters=[z, c, b, av, A, r], variables=[x])

    def forward(self, y, cb: ConstraintBatch, explore=False) -> LayerOut:
        z = torch.sigmoid(y)
        if self.use_cvxpy:
            if self._cvx is None:
                self._cvx = self._cvx_layer(y.shape[-1], cb.P)
            A = cb.onehot().transpose(1, 2)
            (xh,) = self._cvx(z, cb.c, cb.budget, cb.avail, A, cb.cap)
        else:
            xh = dykstra_projection(z, cb, self.iters)
        hard = (xh.detach() > 0.5).float()
        return LayerOut(x=hard + xh - xh.detach(), deploy=hard, aux={"x_proj": xh})


# (v) randomised greedy, score-function gradient (in the spirit of Sakaue 2021:
# sampling over the feasible extension set; scores replace marginal gains).
class RandomisedGreedy(nn.Module):
    stochastic = True

    def __init__(self, eps=0.3):
        super().__init__()
        self.eps = eps

    def forward(self, y, cb: ConstraintBatch, explore=False) -> LayerOut:
        if not explore:
            x = hard_greedy(y.detach(), cb)
            return LayerOut(x=x, deploy=x, logp=torch.zeros(y.shape[0], device=y.device))
        Bsz, N = y.shape
        ar = torch.arange(Bsz, device=y.device)
        b = cb.budget.clone().float(); n = cb.cap.clone().float()
        chosen = torch.zeros_like(y); logp = torch.zeros(Bsz, device=y.device)
        for _ in range(N):
            npart = n.gather(1, cb.part)
            feas = (cb.avail > 0) & (chosen == 0) & (cb.c <= b.unsqueeze(-1) + 1e-9) & (npart >= 1)
            any_f = feas.any(-1)
            if not any_f.any():
                break
            logits = torch.where(feas, y / self.eps, torch.full_like(y, -1e9))
            lp = torch.log_softmax(logits, -1)
            i = torch.distributions.Categorical(logits=logits.detach()).sample()
            logp = logp + torch.where(any_f, lp[ar, i], torch.zeros_like(logp))
            f = any_f.float()
            chosen[ar, i] = torch.maximum(chosen[ar, i], f)
            b = b - cb.c[ar, i] * f
            n[ar, cb.part[ar, i]] -= f
        return LayerOut(x=chosen, deploy=chosen, logp=logp)


# (vi) perturbed optimiser (Berthet et al. 2020) with the hard greedy oracle.
class _PerturbedGreedy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, cb_tensors, eps, M):
        c, budget, part, cap, avail = cb_tensors
        cb = ConstraintBatch(c, budget, part, cap, avail)
        Z = torch.randn((M,) + y.shape, device=y.device)
        X = torch.stack([hard_greedy(y + eps * Z[k], cb) for k in range(M)])
        ctx.save_for_backward(X, Z)
        ctx.eps = eps
        return X.mean(0)

    @staticmethod
    def backward(ctx, g):
        X, Z = ctx.saved_tensors
        gy = ((X * g.unsqueeze(0)).sum(-1, keepdim=True) * Z).mean(0) / ctx.eps
        return gy, None, None, None


class PerturbedOptimiser(nn.Module):
    stochastic = False

    def __init__(self, eps=0.5, M=16):
        super().__init__()
        self.eps, self.M = eps, M

    def forward(self, y, cb: ConstraintBatch, explore=False) -> LayerOut:
        xp = _PerturbedGreedy.apply(y, (cb.c, cb.budget, cb.part, cb.cap, cb.avail),
                                    self.eps, self.M)
        hard = hard_greedy(y.detach(), cb)
        return LayerOut(x=hard + xp - xp.detach(), deploy=hard, aux={"x_pert": xp})


def make_layer(name: str, **kw):
    from .layer import STSG
    table = {
        "stsg": lambda: STSG(**kw),
        "stsg_density": lambda: STSG(key="density", **kw),
        "stsg_softcost": lambda: STSG(soft_sorted_costs=True, **kw),
        "abl_no_matroid_gate": lambda: STSG(matroid_gate=False, **kw),
        "abl_naive_softsort": lambda: STSG(sort="naive", **kw),
        "abl_soft_forward": lambda: STSG(forward_mode="soft", **kw),
        "abl_paper_gate_offset": lambda: STSG(gate_offset=1.0, **kw),
        "penalty": lambda: BernoulliST(),
        "lagrangian": lambda: BernoulliST(),
        "qp_round": lambda: QPProjectRound(**kw),
        "gumbel_topk": lambda: PartTopK(**kw),
        "rand_greedy": lambda: RandomisedGreedy(**kw),
        "perturbed": lambda: PerturbedOptimiser(**kw),
    }
    return table[name]()
