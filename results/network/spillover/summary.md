# Volatility spillover network — Indian IT

## A · Connectedness (full sample)

2020-03-02 → 2026-10-05, 1628 days, VAR(1) on ln RV, 10-day horizon, generalised FEVD.

**Total spillover index: 69.6 %** — share of all forecast-error variance that comes from other nodes.

Share of each row's 10-day volatility forecast-error variance coming from each column (%):

|          |   TCS |   Infosys |   Wipro |   HCLTech |   TechM |   NIFTY IT |   NIFTY 50 |
|:---------|------:|----------:|--------:|----------:|--------:|-----------:|-----------:|
| TCS      |  30.3 |      12.5 |     8.7 |      11.4 |     9.2 |       18.6 |        9.4 |
| Infosys  |  12.7 |      28.2 |     9.4 |      10.5 |     9.6 |       20.7 |        8.8 |
| Wipro    |  10.9 |      11.4 |    32.6 |       9.4 |    10   |       15.5 |       10.1 |
| HCLTech  |  12.4 |      11.8 |     9.2 |      30.1 |    10.8 |       16.8 |        8.7 |
| TechM    |  10.1 |      11.6 |     9.8 |      11.4 |    30.2 |       16.6 |       10.2 |
| NIFTY IT |  14.9 |      17.1 |    10.7 |      12.4 |    11.7 |       23.6 |        9.6 |
| NIFTY 50 |  10.2 |       9.6 |    10.1 |       9   |    10.6 |       12.8 |       37.8 |

| node | FROM others | TO others | NET (TO − FROM) | role |
|---|---|---|---|---|
| NIFTY IT | 76.4 | 101.0 | +24.6 | transmitter |
| Infosys | 71.8 | 74.0 | +2.3 | transmitter |
| TCS | 69.7 | 71.2 | +1.5 | transmitter |
| NIFTY 50 | 62.2 | 56.9 | -5.4 | receiver |
| HCLTech | 69.9 | 64.2 | -5.7 | receiver |
| TechM | 69.8 | 61.9 | -7.9 | receiver |
| Wipro | 67.4 | 58.0 | -9.4 | receiver |

Strongest net pairwise flows (source → target, % of target's forecast-error variance):

| source   | target   |   net_pct |
|:---------|:---------|----------:|
| NIFTY IT | TechM    |      4.88 |
| NIFTY IT | Wipro    |      4.76 |
| NIFTY IT | HCLTech  |      4.44 |
| NIFTY IT | TCS      |      3.72 |
| NIFTY IT | Infosys  |      3.58 |
| NIFTY IT | NIFTY 50 |      3.19 |
| TCS      | Wipro    |      2.26 |
| Infosys  | Wipro    |      1.99 |

Rolling 250-day window: total spillover ranges 58.1–74.5 % (lowest 2022-02, highest 2021-03).

## B · Results-day spillover (does a peer's results day surprise my GARCH?)

excess u = mean (ln RV − ln h_GARCH) on the reporter's results reaction days (excluding the affected firm's own results days) minus days with no known IT results, 2022 →. HCLTech and TechM own results dates are not in the filings table, so their column cannot exclude own-results overlap. exp(excess u) ≈ how many times larger realised variance was than the stock's own GARCH expected, relative to normal days.

Rows = affected stock, columns = company reporting results:

| affected   |   Infosys |   TCS |   Wipro |
|:-----------|----------:|------:|--------:|
| HCLTech    |      0.79 |  0.73 |    0.05 |
| Infosys    |      2.01 |  0.66 |   -0.11 |
| TCS        |      0.26 |  1.09 |   -0.14 |
| TechM      |      0.91 |  0.47 |    0.53 |
| Wipro      |      0.68 |  0.57 |    1.79 |

| reporter   | affected   | own   |   n_days |   mean_u_on |   mean_u_other |   excess_u |      t |   rv_ratio |
|:-----------|:-----------|:------|---------:|------------:|---------------:|-----------:|-------:|-----------:|
| TCS        | TCS        | True  |       20 |       0.573 |         -0.514 |      1.087 |  5.312 |      2.965 |
| TCS        | Infosys    | False |       18 |       0.092 |         -0.573 |      0.664 |  3.048 |      1.943 |
| TCS        | Wipro      | False |       19 |      -0.164 |         -0.732 |      0.568 |  2.672 |      1.765 |
| TCS        | HCLTech    | False |       20 |       0.286 |         -0.447 |      0.733 |  4.763 |      2.081 |
| TCS        | TechM      | False |       20 |       0.007 |         -0.466 |      0.473 |  3.456 |      1.605 |
| Infosys    | TCS        | False |       16 |      -0.253 |         -0.514 |      0.26  |  1.121 |      1.298 |
| Infosys    | Infosys    | True  |       18 |       1.44  |         -0.573 |      2.012 |  6.175 |      7.482 |
| Infosys    | Wipro      | False |       15 |      -0.053 |         -0.732 |      0.679 |  2.422 |      1.972 |
| Infosys    | HCLTech    | False |       18 |       0.345 |         -0.447 |      0.791 |  3.679 |      2.206 |
| Infosys    | TechM      | False |       18 |       0.448 |         -0.466 |      0.914 |  4.398 |      2.494 |
| Wipro      | TCS        | False |       22 |      -0.651 |         -0.514 |     -0.137 | -0.766 |      0.872 |
| Wipro      | Infosys    | False |       20 |      -0.687 |         -0.573 |     -0.114 | -0.655 |      0.892 |
| Wipro      | Wipro      | True  |       23 |       1.055 |         -0.732 |      1.787 |  6.34  |      5.97  |
| Wipro      | HCLTech    | False |       23 |      -0.401 |         -0.447 |      0.045 |  0.24  |      1.046 |
| Wipro      | TechM      | False |       23 |       0.061 |         -0.466 |      0.527 |  3.337 |      1.693 |
