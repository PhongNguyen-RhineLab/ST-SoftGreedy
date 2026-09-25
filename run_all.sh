#!/usr/bin/env bash
# Full pipeline. Each block is independent; comment out what you do not need.
# W = parallel (method, seed) jobs for exp3. Rough cost on one core:
#   exp1 ~ 10 min, exp2 ~ 1-2 h (12 methods x 9 groups x 10 seeds),
#   exp3 ~ 5 h per instance with W=1 (7 methods x 3 seeds x 150 iters).
set -e
export PYTHONPATH=$(pwd)
W=${W:-4}
OUT=${OUT:-results}
mkdir -p $OUT

python -m pytest -q tests/

# Exp 1: theory check (Lemma rank, Thm consistency, gate offset, ablations b and c)
python experiments/exp1_consistency.py --out $OUT

# Exp 2: static coverage, isolates the action layer
python experiments/exp2_static.py --ms 8 16 32 --thetas 2 5 10 --seeds 10 --out $OUT

# Exp 3: main table, then ablations (a)-(d), then ablation (e) without CVaR
python experiments/exp3_bilevel.py --m 8 --theta 5 --T 3 --seeds 0 1 2 --workers $W --out $OUT
python experiments/exp3_bilevel.py --m 8 --theta 5 --T 3 --seeds 0 1 2 --workers $W --out $OUT \
       --methods stsg --ablations
python experiments/exp3_bilevel.py --m 8 --theta 5 --T 3 --seeds 0 1 2 --workers $W --out $OUT \
       --methods stsg --no_cvar

# Exp 4: amortisation bias (small instance), Exp 5: co-location complementarity
python experiments/exp4_amortisation.py --m 6 --T 2 --out $OUT
python experiments/exp5_colocation.py --m 8 --T 3 --out $OUT

python experiments/make_tables.py --res $OUT
