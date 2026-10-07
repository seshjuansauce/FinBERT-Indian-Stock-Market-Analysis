# Sentiment model comparison

Test set: PROVISIONAL — Claude's hidden drafts for 148 of 148 rows

| run | format | seeds | test_macro_f1 | sd | test_acc | sign_flips | f1_co_mention | f1_list | minimal_pairs | dev_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| A_full | full | 1 | 0.757 |  | 0.757 | 9.000 | 0.721 | 0.857 | 0.400 | 0.739 |
| B | pair_tocm | 1 | 0.809 |  | 0.811 | 7.000 | 0.754 | 0.902 | 0.750 | 0.822 |
| C | pair_tocm | 1 | 0.839 |  | 0.838 | 6.000 | 0.781 | 0.929 | 0.800 | 0.827 |

## Paired bootstrap, test macro-F1 difference (mean over seeds)

| comparison | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| B - A_full | +0.052 | [-0.001, +0.107] | 0.029 |
| C - B | +0.031 | [-0.022, +0.088] | 0.134 |
| C - A_full | +0.083 | [+0.032, +0.139] | 0.001 |
