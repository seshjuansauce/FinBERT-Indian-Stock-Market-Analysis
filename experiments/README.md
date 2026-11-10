# Experiments

Two pre-specified tests of one question — **does reading the news with FinBERT make volatility forecasts better than
GARCH?** — at two time scales. All design choices are fixed on the validation period (Jul 2022 – Dec 2023); test results
(2024 →) are reported once.

| | Experiment 1 · daily | Experiment 2 · announcement |
|---|---|---|
| Script | `finbert_vs_garch.py` | `event_reaction.py` |
| Unit | every trading day, TCS | every firm-day with a BSE filing, 6 companies (~4,200) |
| Text | all TCS news headlines that day (HF corpus) | the BSE filing itself (headline + summary) |
| Target | next-day realised variance vs GJR-GARCH | (T1) reaction-day GARCH miss · (T2) P(abnormal 2-day move > 2σ) |
| Baselines | GARCH · + HAR · + article counts | GARCH · + filing metadata (type, timing, company, recent vol) |
| Key test | sentiment beyond article counts | FinBERT beyond filing metadata |
| Inference | Diebold–Mariano, block bootstrap, Model Confidence Set, within-year placebo | date-clustered bootstrap, within type×year placebo |
| Results | `results/experiments/finbert_vs_garch/summary.md` | `results/experiments/event_reaction/summary.md` |

Why both: text is expected to carry information about *how much* a price will react to an event, not which way, and
mostly on the days the event happens (Loughran & McDonald 2011; Cohen, Malloy & Nguyen 2020). A daily test dilutes
that over hundreds of quiet days; the announcement test isolates it.

Reproduce (on the machine that holds `data/`):
```bash
python experiments/finbert_vs_garch.py --placebo 500 --boot 5000
python experiments/event_reaction.py --axiom ~/Desktop/Projects/axiom --placebo 200   # needs data/events/filing_finbert.parquet
```
`data/events/filing_finbert.parquet` = FinBERT-IN (temperature-calibrated) scores of every filing in axiom's
`research/data/filings.parquet` (`python scripts/score_filings.py`).
