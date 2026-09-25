"""Exp 5: 'Supermodular co-location' is stated in the paper as an empirical
property of the simulator. Test it: for site j, background X (site j empty),
a = MCS-A at j, b in {BSS, BESS} at j, compute the second difference
  D2 G = G(X+a+b) - G(X+a) - G(X+b) + G(X)
for G in {-grid cost, served fraction, total operator profit}.
D2 > 0 means complementarity (supermodular) of G on that pair.
Report the fraction of (site, background) samples with D2 > 0 and the mean.
Outcomes use the amortised follower, so the result is conditional on it.
"""
import argparse
import json
import os

import numpy as np
import torch

from agents.follower import FollowerPolicy, train_follower
from experiments.exp4_amortisation import outcome
from env.instance import generate, random_feasible_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--theta", type=float, default=5.0)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--follower_iters", type=int, default=150)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = generate(m=a.m, theta=a.theta, T=a.T, seed=0)
    tag = f"m{a.m}_th{a.theta:g}_T{a.T}_i0"
    fpath = os.path.join(a.out, f"follower_{tag}.pt")
    if os.path.exists(fpath):
        pol = FollowerPolicy(); pol.load_state_dict(torch.load(fpath))
    else:
        pol, _ = train_follower(inst, iters=a.follower_iters, seed=0)
        torch.save(pol.state_dict(), fpath)
    scales = np.exp(0.25 * np.array([-1.15, -0.32, 0.32, 1.15]))
    rng = np.random.default_rng(11)
    G = {"neg_grid": lambda o: -o["grid"], "served_frac": lambda o: o["served_frac"],
         "profit": lambda o: o["profit"]}
    recs = []
    for s in range(a.samples):
        j = int(rng.choice(np.where(inst.od_sites.any(0))[0]))   # sites some OD can use
        X = random_feasible_config(inst, int(rng.integers(inst.T)), rng)
        X[inst.mod_site == j] = 0.0
        ia = np.where((inst.mod_site == j) & (inst.mod_type == 0))[0][0]
        for tb, bname in ((2, "BSS"), (3, "BESS")):
            ib = np.where((inst.mod_site == j) & (inst.mod_type == tb))[0][0]
            xs = {}
            for key, add in (("0", []), ("a", [ia]), ("b", [ib]), ("ab", [ia, ib])):
                x = X.copy(); x[add] = 1.0
                xs[key] = outcome(inst, pol, x, scales)
            rec = {"sample": s, "site": j, "pair": f"MCS+{bname}"}
            for gname, g in G.items():
                rec[gname] = g(xs["ab"]) - g(xs["a"]) - g(xs["b"]) + g(xs["0"])
            recs.append(rec)
        print(f"sample {s} site {j}: " + " ".join(f"{r['pair']} dgrid {r['neg_grid']:.3f}" for r in recs[-2:]), flush=True)
    summ = {}
    for pair in ("MCS+BSS", "MCS+BESS"):
        rr = [r for r in recs if r["pair"] == pair]
        summ[pair] = {g: {"frac_pos": float(np.mean([r[g] > 1e-9 for r in rr])),
                          "frac_neg": float(np.mean([r[g] < -1e-9 for r in rr])),
                          "mean": float(np.mean([r[g] for r in rr]))} for g in G}
    print(json.dumps(summ, indent=1))
    json.dump({"records": recs, "summary": summ}, open(os.path.join(a.out, f"exp5_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
