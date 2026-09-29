#!/usr/bin/env bash
# Full pipeline. Each block is independent; comment out what you do not need.
# W = parallel (method, seed) jobs for exp3. Rough cost on one core:
#   exp1 ~ 10 min, exp2 ~ 1-2 h (12 methods x 9 groups x 10 seeds),
#   exp3 at m=16 ~ 15-25 min per (method, seed) run; 130 runs ~ 35-55 h worker-time, divide by W.
set -e
export PYTHONPATH=$(pwd)
W=${W:-4}
OUT=${OUT:-results_m16}
mkdir -p $OUT

python -m pytest -q tests/

# Exp 1: theory check (Lemma rank, Thm consistency, gate offset, ablations b and c)
python experiments/exp1_consistency.py --out $OUT

# Exp 2: static coverage, isolates the action layer
python experiments/exp2_static.py --ms 8 16 32 --thetas 2 5 10 --seeds 10 --out $OUT

# Exp 3 on m = 16 (the m = 8 main instance saturates coverage after stage 2, see README).
# Step 0 trains the follower once and prints the constraint landscape of the heuristic plans;
# check that the coverage greedy violates the CMDP and that some heuristic satisfies it
# BEFORE launching the learners. Change --beta_* here if not, and keep them fixed afterwards.
# Runs are merged into $OUT/exp3_<tag>.json; stored (method, seed) pairs are skipped.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SEEDS="0 1 2 3 4 5 6 7 8 9"
E3="python experiments/exp3_bilevel.py --m 16 --theta 5 --T 3 --out $OUT"
$E3 --probe
$E3 --seeds $SEEDS --workers $W
$E3 --seeds $SEEDS --workers $W --methods stsg --ablations
$E3 --seeds $SEEDS --workers $W --methods stsg --no_cvar

# Exp 4: amortisation bias (small instance), Exp 5: co-location complementarity
python experiments/exp4_amortisation.py --m 6 --T 2 --out $OUT
python experiments/exp5_colocation.py --m 8 --T 3 --out $OUT

python experiments/make_tables.py --res $OUT
