"""pytest -q tests/  --  checks of Proposition feasible, ST semantics, Lemma rank,
Theorem consistency, and the naive-softsort remark."""
import math
import numpy as np
import torch

from stsg.layer import STSG, ConstraintBatch, hard_greedy, consistency_diagnostics, naive_softsort, softsort
from stsg.baselines import make_layer
from stsg.coverage import static_key_greedy, rerank_greedy, milp_multistage, coverage_np, feasible_np
from env.instance import generate


def rand_cb(B=64, N=40, P=10, seed=0, theta=5.0):
    g = torch.Generator().manual_seed(seed)
    c = 1 + (theta - 1) * torch.rand(B, N, generator=g)
    part = torch.randint(0, P, (B, N), generator=g)
    cap = torch.randint(1, 4, (B, P), generator=g).float()
    avail = (torch.rand(B, N, generator=g) > 0.1).float()
    budget = 2 * theta + 10 * torch.rand(B, generator=g)
    return ConstraintBatch(c, budget, part, cap, avail)


def test_hard_feasibility_all_temperatures():
    cb = rand_cb()
    for tau1 in (1e-3, 0.1, 10.0):
        for tau2 in (1e-3, 0.1, 10.0):
            for key in ("score", "density"):
                y = torch.randn(64, 40, requires_grad=True)
                out = STSG(tau1, tau2, key=key)(y, cb)
                assert set(out.deploy.unique().tolist()) <= {0.0, 1.0}
                assert cb.violation(out.deploy)["infeasible"].sum() == 0
                assert torch.allclose(out.x.detach(), out.deploy)


def test_st_gradient_nonzero():
    cb = rand_cb()
    y = torch.randn(64, 40, requires_grad=True)
    out = STSG(0.1, 0.05)(y, cb)
    (out.x * torch.randn_like(out.x)).sum().backward()
    assert y.grad.abs().sum() > 0


def test_torch_greedy_matches_numpy():
    cb = rand_cb(B=8)
    y = torch.randn(8, 40)
    xt = hard_greedy(y, cb)
    for b in range(8):
        xn = static_key_greedy(y[b].numpy(), cb.c[b].numpy(), cb.budget[b].item(),
                               cb.part[b].numpy(), cb.cap[b].numpy(), cb.avail[b].numpy())
        assert np.allclose(xn, xt[b].numpy())


def test_rank_lemma_and_consistency():
    cb = rand_cb(B=32)
    y = torch.randn(32, 40) * 5
    d = consistency_diagnostics(y, cb, tau1=1e-3, tau2=1e-3)
    assert (d["rank_err"] <= d["rank_bound"] + 1e-6).all()
    ok = d["precond_ok"]
    assert (d["err_l1"][ok] <= d["bound"][ok] + 1e-6).all()


def test_naive_softsort_is_identity_limit():
    s = torch.randn(4, 12)
    P, _ = naive_softsort(s, 1e-4)
    assert torch.allclose(P, torch.eye(12).expand(4, -1, -1), atol=1e-3)
    P2, perm = softsort(s, 1e-4)
    assert torch.allclose(P2, torch.nn.functional.one_hot(perm, 12).float(), atol=1e-3)


def test_feasible_by_construction_baselines():
    cb = rand_cb(B=16)
    y = torch.randn(16, 40, requires_grad=True)
    for name in ("rand_greedy", "perturbed"):
        out = make_layer(name)(y, cb, explore=True)
        assert cb.violation(out.deploy)["infeasible"].sum() == 0


def test_milp_dominates_greedy():
    inst = generate(m=6, theta=5, T=2, seed=3)
    val, X, _ = milp_multistage(inst.A, inst.w, inst.cost, inst.stage_budget, inst.mod_site, inst.cap, 1.0)
    budget = inst.stage_budget.sum()
    ones = np.ones(inst.N)
    g = rerank_greedy(inst.A, inst.w, inst.cost, budget, inst.mod_site, inst.cap, ones, rule="best")
    assert feasible_np(X[-1], inst.cost, budget, inst.mod_site, inst.cap, ones)
    # final-stage value of MILP >= greedy at the final stage? only the sum is optimal; check single stage
    v1, X1, _ = milp_multistage(inst.A, inst.w, inst.cost, [budget], inst.mod_site, inst.cap, 1.0)
    assert v1 + 1e-6 >= coverage_np(g, inst.A, inst.w)
