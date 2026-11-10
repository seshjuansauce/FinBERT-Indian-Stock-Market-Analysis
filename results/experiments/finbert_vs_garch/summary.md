# Does FinBERT improve GARCH? — TCS

Data: walk-forward forecasts, validation 2022-07-01 → 2023-12-31, test 2024-01-01 → 2026-03-30 (end of news corpus). Loss: QLIKE (lower is better). Negative differences favour the first model.

## Models

| period     | model                                |   n |   qlike |   vs_M0_pct |   rmse_lnRV |   mz_r2 |
|:-----------|:-------------------------------------|----:|--------:|------------:|------------:|--------:|
| validation | M0 GJR-GARCH                         | 369 |  0.3394 |      0      |      0.9429 |  0.0688 |
| validation | M1 +HAR+market                       | 369 |  0.3095 |     -8.8187 |      0.8571 |  0.0793 |
| validation | M2 +news volume                      | 369 |  0.3005 |    -11.4711 |      0.8424 |  0.0907 |
| validation | M3 +FinBERT-IN sentiment (no volume) | 369 |  0.3149 |     -7.2044 |      0.8654 |  0.0688 |
| validation | M4 +volume +FinBERT-IN               | 369 |  0.316  |     -6.905  |      0.8623 |  0.0774 |
| validation | M4a +volume +ProsusAI FinBERT        | 369 |  0.312  |     -8.0747 |      0.8581 |  0.0768 |
| test       | M0 GJR-GARCH                         | 548 |  0.4398 |      0      |      0.9872 |  0.0863 |
| test       | M1 +HAR+market                       | 548 |  0.4377 |     -0.4646 |      0.8993 |  0.0969 |
| test       | M2 +news volume                      | 548 |  0.4273 |     -2.8254 |      0.8893 |  0.112  |
| test       | M3 +FinBERT-IN sentiment (no volume) | 548 |  0.4389 |     -0.19   |      0.8924 |  0.0999 |
| test       | M4 +volume +FinBERT-IN               | 548 |  0.4331 |     -1.5209 |      0.8896 |  0.1068 |
| test       | M4a +volume +ProsusAI FinBERT        | 548 |  0.433  |     -1.5491 |      0.8873 |  0.109  |

## The questions (mean daily QLIKE difference, 95% block-bootstrap CI, Diebold–Mariano)

| period     | question                                   | model                                | reference                     |   mean_dQLIKE |     pct |   ci95_lo |   ci95_hi |      dm |      p | verdict                   |
|:-----------|:-------------------------------------------|:-------------------------------------|:------------------------------|--------------:|--------:|----------:|----------:|--------:|-------:|:--------------------------|
| validation | Q1 full model vs GARCH                     | M4 +volume +FinBERT-IN               | M0 GJR-GARCH                  |       -0.0234 | -6.905  |   -0.0523 |    0.0117 | -1.4102 | 0.1585 | no significant difference |
| validation | Q2 beyond past volatility (HAR)            | M4 +volume +FinBERT-IN               | M1 +HAR+market                |        0.0065 |  2.0987 |   -0.0175 |    0.0329 |  0.5185 | 0.6041 | no significant difference |
| validation | Q3 FinBERT sentiment beyond article counts | M4 +volume +FinBERT-IN               | M2 +news volume               |        0.0155 |  5.1577 |    0.0032 |    0.0324 |  2.1051 | 0.0353 | worse                     |
| validation | Q3b sentiment alone, over HAR              | M3 +FinBERT-IN sentiment (no volume) | M1 +HAR+market                |        0.0055 |  1.7705 |   -0.0097 |    0.0236 |  0.6439 | 0.5196 | no significant difference |
| validation | Q4 domain-adapted vs off-the-shelf FinBERT | M4 +volume +FinBERT-IN               | M4a +volume +ProsusAI FinBERT |        0.004  |  1.2724 |   -0.0053 |    0.0184 |  0.5995 | 0.5488 | no significant difference |
| validation | (ref) article counts over HAR              | M2 +news volume                      | M1 +HAR+market                |       -0.009  | -2.9089 |   -0.0334 |    0.0131 | -0.7727 | 0.4397 | no significant difference |
| test       | Q1 full model vs GARCH                     | M4 +volume +FinBERT-IN               | M0 GJR-GARCH                  |       -0.0067 | -1.5209 |   -0.0389 |    0.0259 | -0.4001 | 0.6891 | no significant difference |
| test       | Q2 beyond past volatility (HAR)            | M4 +volume +FinBERT-IN               | M1 +HAR+market                |       -0.0046 | -1.0612 |   -0.0346 |    0.0234 | -0.3102 | 0.7564 | no significant difference |
| test       | Q3 FinBERT sentiment beyond article counts | M4 +volume +FinBERT-IN               | M2 +news volume               |        0.0057 |  1.3425 |   -0.0037 |    0.0162 |  1.1286 | 0.2591 | no significant difference |
| test       | Q3b sentiment alone, over HAR              | M3 +FinBERT-IN sentiment (no volume) | M1 +HAR+market                |        0.0012 |  0.2758 |   -0.0134 |    0.0157 |  0.1627 | 0.8708 | no significant difference |
| test       | Q4 domain-adapted vs off-the-shelf FinBERT | M4 +volume +FinBERT-IN               | M4a +volume +ProsusAI FinBERT |        0.0001 |  0.0287 |   -0.0043 |    0.0048 |  0.0545 | 0.9565 | no significant difference |
| test       | (ref) article counts over HAR              | M2 +news volume                      | M1 +HAR+market                |       -0.0104 | -2.3718 |   -0.0367 |    0.0128 | -0.8079 | 0.4191 | no significant difference |

## Placebo (500 within-year shuffles of the FinBERT-IN sentiment columns)

- validation: real M4 QLIKE 0.3160; placebo mean 0.3074 (5–95%: 0.2990–0.3149); share of placebos at least as good = **0.96**
- test: real M4 QLIKE 0.4331; placebo mean 0.4304 (5–95%: 0.4238–0.4371); share of placebos at least as good = **0.78**

## Model Confidence Set (test, 90%): ['M0 GJR-GARCH', 'M1 +HAR+market', 'M2 +news volume', 'M3 +FinBERT-IN sentiment (no volume)', 'M4 +volume +FinBERT-IN', 'M4a +volume +ProsusAI FinBERT']  ·  eliminated: []  ·  p-values: M4a 0.74, M4 0.74, M3 0.74, M1 0.74, M0 0.74, M2 1.00

## Where (test period)

| subset                               |   n |     M0 |     M1 |     M2 |     M4 |   M4_vs_M2_pct |   dm_M4_vs_M2_p |
|:-------------------------------------|----:|-------:|-------:|-------:|-------:|---------------:|----------------:|
| all test days                        | 548 | 0.4398 | 0.4377 | 0.4273 | 0.4331 |         1.3425 |          0.2591 |
| results-news days                    |  20 | 0.7784 | 1.1183 | 0.5165 | 0.5012 |        -2.964  |        nan      |
| news days (≥1 article)               | 418 | 0.4178 | 0.4294 | 0.4106 | 0.4158 |         1.2786 |          0.3714 |
| no-news days                         | 130 | 0.5103 | 0.4644 | 0.4812 | 0.4885 |         1.5176 |          0.4935 |
| strongly negative news (sent < −0.3) |  97 | 0.4854 | 0.4707 | 0.5053 | 0.5275 |         4.4022 |          0.2789 |
| strongly positive news (sent > 0.3)  | 137 | 0.3096 | 0.2713 | 0.2692 | 0.2778 |         3.1591 |          0.2231 |
| year 2024                            | 244 | 0.3601 | 0.3959 | 0.3828 | 0.3755 |        -1.9148 |          0.0926 |
| year 2025                            | 246 | 0.4569 | 0.4165 | 0.3918 | 0.4037 |         3.0335 |          0.1727 |
| year 2026                            |  58 | 0.7021 | 0.7037 | 0.7652 | 0.7998 |         4.5248 |          0.1265 |
