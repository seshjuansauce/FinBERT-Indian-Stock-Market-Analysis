# Temperature scaling (T fitted on 68 dev headlines; test = 148 headlines, labels: claude_provisional)

| model | T | test NLL | test ECE | test Brier | mean confidence | accuracy (unchanged) | share \|s\| > 0.9 |
|---|---|---|---|---|---|---|---|
| A | 1.80 | 0.694 → 0.641 | 0.131 → 0.070 | 0.386 → 0.359 | 0.886 → 0.727 | 0.757 | 0.28 → 0.00 |
| B | 2.79 | 1.135 → 0.575 | 0.173 → 0.114 | 0.367 → 0.310 | 0.980 → 0.845 | 0.811 | 0.60 → 0.00 |
| C | 2.52 | 0.859 → 0.496 | 0.152 → 0.080 | 0.313 → 0.267 | 0.978 → 0.854 | 0.838 | 0.57 → 0.00 |
