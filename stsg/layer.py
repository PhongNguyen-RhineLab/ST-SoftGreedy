"""Straight-through soft-greedy (STSG) layer for partition-matroid ∩ knapsack.

Follows Section "The Straight-Through Soft-Greedy Layer" of the paper.
Shapes: batch B, ground set N (modules), parts P (sites).

Paper -> code map
  eq. (softsort)   -> softsort()
  Remark softsort  -> naive_softsort()          (ablation b)
  eq. (gate)       -> soft_gate_scan()
  Algorithm 1      -> hard_greedy()
  eq. (st)         -> STSG.forward, forward_mode="st"
  Lemma rank / Def margin / Thm consistency -> consistency_diagnostics()

NOTE on the matroid gate offset. The paper writes sigma((n-1)/tau2). The hard
test is n >= 1 with integer n, so a part with exactly one free slot sits on the
gate boundary: sigma(0) = 1/2 and |n-1| = 0, which violates the delta-margin of
Definition 7 on essentially every trajectory that fills a part. Using
sigma((n-1/2)/tau2) keeps the same hard decision and gives margin 1/2 for free.
gate_offset=0.5 is the default here; gate_offset=1.0 reproduces the paper text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# constraint container
# --------------------------------------------------------------------------
@dataclass
class ConstraintBatch:
    """X_t = {S : c(S) <= budget, |S ∩ V_p| <= cap_p, S ⊆ avail}."""
    c: torch.Tensor        # (B,N) costs > 0
    budget: torch.Tensor   # (B,)  residual budget B_t
    part: torch.Tensor     # (B,N) long, part index in [0,P)
    cap: torch.Tensor      # (B,P) residual capacity r_p
    avail: torch.Tensor    # (B,N) 1 if element may be chosen

    @property
    def P(self) -> int:
        return self.cap.shape[-1]

    def onehot(self) -> torch.Tensor:
        return F.one_hot(self.part, self.P).float()

    def violation(self, x: torch.Tensor, tol: float = 1e-6) -> dict:
        """Violation of a (binary or fractional) action x (B,N)."""
        over_b = F.relu((self.c * x).sum(-1) - self.budget)
        load = torch.einsum("bn,bnp->bp", x, self.onehot())
        over_p = F.relu(load - self.cap).sum(-1)
        over_a = (x * (1 - self.avail)).sum(-1)
        total = over_b + over_p + over_a
        return {"budget": over_b, "part": over_p, "avail": over_a,
                "total": total, "infeasible": (total > tol).float()}

    def to(self, dev):
        return ConstraintBatch(*(t.to(dev) for t in
                                 (self.c, self.budget, self.part, self.cap, self.avail)))


@dataclass
class LayerOut:
    x: torch.Tensor                    # action used in the training graph
    deploy: torch.Tensor               # binary action executed by the environment
    logp: torch.Tensor | None = None   # score-function layers only
    aux: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Step 1: ranking
# --------------------------------------------------------------------------
def ranking_key(y: torch.Tensor, c: torch.Tensor, key: str) -> torch.Tensor:
    if key == "score":
        return y
    if key == "density":           # Algorithm 1 alternative key s_i = y_i / c_i
        return y / c
    raise ValueError(key)


def softsort(s: torch.Tensor, tau1: float):
    """eq. (softsort): row k compares against the k-th order statistic."""
    s_sorted, perm = torch.sort(s, dim=-1, descending=True)
    logits = -(s_sorted.unsqueeze(-1) - s.unsqueeze(-2)).abs() / tau1
    return torch.softmax(logits, dim=-1), perm


def naive_softsort(s: torch.Tensor, tau1: float):
    """Remark softsort: softmax_row(-|s_i - s_j|/tau). Tends to I, does not sort."""
    logits = -(s.unsqueeze(-1) - s.unsqueeze(-2)).abs() / tau1
    perm = torch.argsort(s, dim=-1, descending=True)
    return torch.softmax(logits, dim=-1), perm


def sinkhorn(P: torch.Tensor, iters: int) -> torch.Tensor:
    for _ in range(iters):
        P = P / P.sum(-2, keepdim=True).clamp_min(1e-12)
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
    return P


# --------------------------------------------------------------------------
# Algorithm 1: hard greedy (forward pass)
# --------------------------------------------------------------------------
@torch.no_grad()
def hard_greedy(s, cb: ConstraintBatch, return_trace: bool = False):
    """Exact greedy in the order of s (decreasing). Feasible by construction.

    trace (optional): slacks along the hard trajectory for Definition 7,
      budget_slack[k] = b_k - c^(k), part_slack[k] = n_k^(p(k)) (raw count).
    """
    Bsz, N = s.shape
    perm = torch.argsort(s, dim=-1, descending=True)
    ar = torch.arange(Bsz, device=s.device)
    b = cb.budget.clone().float()
    n = cb.cap.clone().float()
    x = torch.zeros_like(s)
    bs, ps, av = [], [], []
    for k in range(N):
        i = perm[:, k]
        ci, pi, ai = cb.c[ar, i], cb.part[ar, i], cb.avail[ar, i]
        nk = n[ar, pi]
        ok = (ci <= b + 1e-9) & (nk >= 1 - 1e-9) & (ai > 0)
        if return_trace:
            bs.append(b - ci); ps.append(nk); av.append(ai)
        okf = ok.float()
        x[ar, i] = okf
        b = b - ci * okf
        n[ar, pi] = nk - okf
    if not return_trace:
        return x
    return x, {"perm": perm, "budget_slack": torch.stack(bs, 1),
               "part_slack": torch.stack(ps, 1), "avail_sorted": torch.stack(av, 1)}


# --------------------------------------------------------------------------
# Step 2: gated soft greedy scan, eq. (gate)
# --------------------------------------------------------------------------
def soft_gate_scan(c_s, onehot_s, avail_s, budget, cap, tau2,
                   matroid_gate=True, gate_offset=0.5):
    """Sequential O(N) scan (does not parallelise, see Complexity paragraph).

    c_s (B,N) costs in sorted order, onehot_s (B,N,P) part membership in sorted
    order (hard one-hot or soft rows of P^s @ onehot), avail_s (B,N).
    """
    Bsz, N = c_s.shape
    b = budget.float()
    n = cap.float()
    ms = []
    for k in range(N):
        g = torch.sigmoid((b - c_s[:, k]) / tau2)
        if matroid_gate:
            nk = (n * onehot_s[:, k]).sum(-1)
            g = g * torch.sigmoid((nk - gate_offset) / tau2)
        mk = g * avail_s[:, k]
        b = b - c_s[:, k] * mk
        n = n - onehot_s[:, k] * mk.unsqueeze(-1)
        ms.append(mk)
    return torch.stack(ms, 1)


# --------------------------------------------------------------------------
# Step 3: the layer
# --------------------------------------------------------------------------
class STSG(nn.Module):
    """Straight-through soft-greedy layer.

    forward_mode "st"   : value = hard greedy, Jacobian = d x_tilde / d y   (ours)
    forward_mode "soft" : value = x_tilde, deploy = round(x_tilde)          (ablation c)
    sort "softsort" | "naive"                                              (ablation b)
    matroid_gate False                                                      (ablation a)
    key "score" | "density"                                                 (ablation d)
    soft_sorted_costs: False follows the paper (c^(k) = c_pi(k), hard index),
      so m carries no gradient and d x_tilde / d y flows through P^s only.
      True uses c_tilde = P^s c so m depends on y too (design option, not in paper).
    """
    stochastic = False

    def __init__(self, tau1=0.1, tau2=0.05, key="score", matroid_gate=True,
                 sort="softsort", forward_mode="st", gate_offset=0.5,
                 soft_sorted_costs=False, sinkhorn_iters=0):
        super().__init__()
        self.tau1, self.tau2, self.key = tau1, tau2, key
        self.matroid_gate, self.sort, self.forward_mode = matroid_gate, sort, forward_mode
        self.gate_offset, self.soft_sorted_costs = gate_offset, soft_sorted_costs
        self.sinkhorn_iters = sinkhorn_iters

    def relaxed(self, y, cb: ConstraintBatch):
        s = ranking_key(y, cb.c, self.key)
        P, perm = (softsort if self.sort == "softsort" else naive_softsort)(s, self.tau1)
        if self.sinkhorn_iters:
            P = sinkhorn(P, self.sinkhorn_iters)
        oh = cb.onehot()
        if self.soft_sorted_costs:
            c_s = torch.einsum("bki,bi->bk", P, cb.c)
            oh_s = torch.einsum("bki,bip->bkp", P, oh)
            av_s = torch.einsum("bki,bi->bk", P, cb.avail)
        else:
            c_s = cb.c.gather(1, perm)
            oh_s = oh.gather(1, perm.unsqueeze(-1).expand(-1, -1, oh.shape[-1]))
            av_s = cb.avail.gather(1, perm)
        m = soft_gate_scan(c_s, oh_s, av_s, cb.budget, cb.cap, self.tau2,
                           self.matroid_gate, self.gate_offset)
        x_tilde = torch.einsum("bki,bk->bi", P, m)       # (P^s)^T m
        return x_tilde, {"P": P, "perm": perm, "m": m, "s": s}

    def forward(self, y, cb: ConstraintBatch, explore: bool = False) -> LayerOut:
        x_tilde, aux = self.relaxed(y, cb)
        if self.forward_mode == "st":
            x_bar = hard_greedy(aux["s"].detach(), cb)
            x = x_bar + x_tilde - x_tilde.detach()
            deploy = x_bar
        elif self.forward_mode == "soft":
            x = x_tilde
            deploy = (x_tilde.detach() > 0.5).float()
        else:
            raise ValueError(self.forward_mode)
        aux["x_tilde"] = x_tilde
        return LayerOut(x=x, deploy=deploy, aux=aux)


# --------------------------------------------------------------------------
# diagnostics for Lemma rank / Lemma gate / Theorem consistency
# --------------------------------------------------------------------------
@torch.no_grad()
def consistency_diagnostics(y, cb: ConstraintBatch, tau1, tau2, key="score",
                            gate_offset=0.5, sort="softsort"):
    """Measured quantities and the bounds stated in the paper, per batch row."""
    layer = STSG(tau1, tau2, key=key, gate_offset=gate_offset, sort=sort)
    x_tilde, aux = layer.relaxed(y, cb)
    s = aux["s"]
    x_bar, tr = hard_greedy(s, cb, return_trace=True)
    Bsz, N = s.shape
    P_pi = F.one_hot(tr["perm"], N).float()
    rank_err = (aux["P"] - P_pi).abs().sum(-1).amax(-1)            # ||.||_inf (max row sum)
    col_dev = (aux["P"].sum(-2) - 1).abs().amax(-1)
    s_sorted = torch.sort(s, -1, descending=True).values
    gamma = (s_sorted[:, :-1] - s_sorted[:, 1:]).amin(-1)
    av = tr["avail_sorted"] > 0
    big = torch.full_like(tr["budget_slack"], float("inf"))
    d_b = torch.where(av, tr["budget_slack"].abs(), big).amin(-1)
    d_p = torch.where(av, (tr["part_slack"] - gate_offset).abs(), big).amin(-1)
    delta = torch.minimum(d_b, d_p)
    lam = cb.c.amax(-1).clamp_min(1.0)
    rho = torch.exp(-delta / (2 * tau2))
    precond = (N * lam * rho / tau2) <= math.log(2)
    err = (x_tilde - x_bar).abs().sum(-1)
    bound = 4 * N * rho + 2 * N ** 2 * torch.exp(-gamma / tau1)
    return {"err_l1": err, "bound": bound, "precond_ok": precond,
            "rank_err": rank_err, "rank_bound": 2 * (N - 1) * torch.exp(-gamma / tau1),
            "col_dev": col_dev, "gamma": gamma, "delta": delta,
            "delta_part_paper": torch.where(av, (tr["part_slack"] - 1).abs(), big).amin(-1)}
