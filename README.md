# STSG-MCS: thực nghiệm cho paper AAMAS 2027

Straight-through soft-greedy layer trên partition matroid ∩ knapsack, gắn vào
C-BMDP đồng quy hoạch hành lang MCS-BSS. Code bám theo bản LaTeX; bảng ánh xạ
ở đầu `stsg/layer.py`.

## Cài đặt và chạy

    pip install -r requirements.txt
    export PYTHONPATH=$(pwd)
    python -m pytest -q tests/          # 7 test, gồm Prop. feasible và Thm consistency
    W=8 bash run_all.sh                 # toàn bộ, kết quả trong results/, bảng trong results/tables.tex

## Cấu trúc

    stsg/layer.py        STSG: softsort, gate scan, hard greedy (Alg. 1), straight-through, chẩn đoán lý thuyết
    stsg/baselines.py    penalty, Lagrangian, QP projection + round, Gumbel top-k, randomised greedy, perturbed
    stsg/coverage.py     f(S) submodular, greedy re-rank, MILP nhiều giai đoạn (HiGHS)
    env/instance.py      bộ sinh hành lang: m site, N = 4m module (MCS-A, MCS-B, BSS, BESS), theta chính xác
    env/ue.py            UE Beckmann, Frank-Wolfe + line search
    env/operations.py    trò chơi follower: giá, BSS, BESS, lưới, trễ, SoH
    env/leader_env.py    leader MDP T giai đoạn
    agents/follower.py   IPPO amortised có điều kiện theo X
    agents/leader.py     actor-critic có ràng buộc, dual ascent, CVaR (Rockafellar-Uryasev)
    metrics/nashconv.py  NashConv với tập lệch giá hằng (cận dưới)
    experiments/         exp1..exp5, make_tables.py

## Thí nghiệm và mục paper tương ứng

| Script | Nội dung | Mục paper |
|---|---|---|
| exp1_consistency | sai số ‖x̃ − x̄‖₁ theo τ, cận Lemma rank / Thm consistency, offset gate, ablation b c | Sec. layer |
| exp2_static | tối ưu f theo từng instance qua từng lớp, gap so với MILP, tỉ lệ vi phạm | bảng phụ |
| exp3_bilevel | bảng chính: vi phạm, coverage gap, return, CVaR, thời gian; `--ablations`, `--no_cvar` | Table tab:main |
| exp4_amortisation | amortised vs retrain theo từng X: outcome và giá trị biên của module | Amortisation |
| exp5_colocation | sai phân bậc hai MCS+BSS, MCS+BESS | Supermodular co-location |

## Những điểm paper phải khớp với code

1. Gate matroid dùng σ((n − 1/2)/τ₂). Bản paper σ((n − 1)/τ₂) cho δ = 0 khi part còn 1 slot, Thm consistency không áp dụng được. Tái hiện bằng `abl_paper_gate_offset` / cột offset 1 của exp1.
2. Mặc định c^(k) = c_π(k) như paper, nên m không phụ thuộc ŷ. `stsg_softcost` dùng c̃ = P^s c.
3. Coverage gap chỉ đo thành phần phủ; MILP không mô hình follower game.
4. NashConv là cận dưới (chỉ lệch giá hằng).
5. Exp4 đo amortisation bias của outcome và giá trị biên, không phải hypergradient.
6. Demand hoàn toàn tổng hợp, không có OD matrix công bố.
7. Leader thấy follower qua K = 8 phân vị lognormal cố định của demand (common random numbers), kết quả follower được cache theo (X, k).
8. Hằng số Wang 2011 trong `env/instance.py` phải đối chiếu với bài gốc (đánh dấu VERIFY).
9. Đơn giản hóa: một điểm dừng mỗi chuyến, không V2G, không PV, overflow tại trạm tính là unserved. Frank-Wolfe dừng ở 80 vòng, relative gap vài phần trăm với BPR bậc 4.

## Tham số chính

- Instance: `generate(m, theta, T, seed, budget_frac=0.2, cap_per_site=2, window_lo=0.5)`, L = 100m + 1.9R.
- STSG: `tau1=0.1, tau2=0.05`, đổi qua `LeaderConfig.layer_kw`.
- Leader: `LeaderConfig` (iters 150, 16 episode/iter, gamma 0.95, alpha CVaR 0.9, beta theo bội số chi phí tham chiếu).
