# Can FinBERT predict the initial reaction to a BSE announcement?

4,231 firm-days with filings, 6 companies, 2021-01 → 2026-06. Out-of-sample years 2023–2026 (expanding walk-forward).

Events by type: press 1084, general 1011, procedural 732, other 541, analyst 380, results 268, mna 109, mgmt 59, payout 47

## T1 · Reaction-day volatility beyond GARCH  (target u = ln RV_R − ln h_R; lower MSE / QLIKE is better)

| model               |    n |   mse_u |   r2_oos_vs_B0 |   qlike_RV |
|:--------------------|-----:|--------:|---------------:|-----------:|
| B0                  | 2631 |  1.0834 |         0      |     0.5912 |
| B1 metadata         | 2631 |  0.8511 |         0.2144 |     0.5218 |
| B2 metadata+FinBERT | 2631 |  0.8344 |         0.2298 |     0.5109 |
| B3 company+FinBERT  | 2631 |  0.859  |         0.2072 |     0.5474 |

B2 − B1 (does FinBERT add to metadata?): ΔMSE = -0.0167, 95% date-cluster CI [-0.0296, -0.0050]

Results days only:

| model               |   n |   mse_u (results days) |
|:--------------------|----:|-----------------------:|
| B0                  | 169 |                 1.8565 |
| B1 metadata         | 169 |                 1.4927 |
| B2 metadata+FinBERT | 169 |                 1.291  |
| B3 company+FinBERT  | 169 |                 1.7781 |

FinBERT's contribution (B2 vs B1) by subset, with 95% date-cluster CIs:

| subset                 |    n |   B1_mse |   B2_mse |   change_pct |   ci95_lo |   ci95_hi |
|:-----------------------|-----:|---------:|---------:|-------------:|----------:|----------:|
| all                    | 2631 |   0.8511 |   0.8344 |      -1.9631 |   -0.0296 |   -0.005  |
| results days           |  169 |   1.4927 |   1.291  |     -13.5132 |   -0.3826 |   -0.0255 |
| excluding results days | 2462 |   0.8071 |   0.8031 |      -0.4967 |   -0.0082 |    0.0002 |
| after-close filings    | 1856 |   0.8066 |   0.7837 |      -2.84   |   -0.0397 |   -0.0072 |
| intraday filings       |  775 |   0.9577 |   0.9559 |      -0.1945 |   -0.0152 |    0.011  |
| year 2023              |  814 |   0.8337 |   0.8203 |      -1.6005 |   -0.0347 |    0.0053 |
| year 2024              |  871 |   0.8857 |   0.8705 |      -1.7225 |   -0.0344 |    0.0027 |
| year 2025              |  669 |   0.7381 |   0.724  |      -1.9027 |   -0.0465 |    0.013  |
| year 2026              |  277 |   1.0668 |   1.0292 |      -3.525  |   -0.0773 |   -0.0001 |

## T2 · Probability of an abnormally large move  (|CAR(0,1)| > 2σ̂; lower Brier / log loss, higher AUC is better)

| model               |    n |   base_rate |   brier |   brier_skill_vs_B0 |   log_loss |    auc |
|:--------------------|-----:|------------:|--------:|--------------------:|-----------:|-------:|
| B0                  | 2061 |      0.0665 |  0.0622 |              0      |     0.2456 | 0.5    |
| B1 metadata         | 2061 |      0.0665 |  0.0584 |              0.0603 |     0.2219 | 0.7404 |
| B2 metadata+FinBERT | 2061 |      0.0665 |  0.0585 |              0.0593 |     0.2213 | 0.7452 |
| B3 company+FinBERT  | 2061 |      0.0665 |  0.0609 |              0.021  |     0.2425 | 0.596  |

B2 − B1 Brier: +0.00006, 95% date-cluster CI [-0.00065, +0.00078]

## Placebos (400 shuffles each of the FinBERT features)

- P1 type×year · T1: real MSE 0.8344, placebo mean 0.8546, share at least as good **0.000** · T2: real Brier 0.05849, placebo mean 0.05871, share **0.198**
- P2 type×company · T1: real MSE 0.8344, placebo mean 0.8548, share at least as good **0.000** · T2: real Brier 0.05849, placebo mean 0.05874, share **0.140**

## Direction vs magnitude (out-of-sample years, rank correlations)

- tone s vs CAR(0,1) (direction): ρ = +0.007 (p = 0.745); results days only ρ = -0.019 (p = 0.821, n = 139)
- polarity 1 − P(neutral) vs |CAR|/σ̂ (magnitude): ρ = +0.002 (p = 0.921)
