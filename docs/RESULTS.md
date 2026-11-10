# FinBERT and the volatility reaction to BSE announcements — write-up

Sukrit Gupta · November 2026 · code and data: this repository · verify: `python experiments/verify_claims.py`

## Abstract

We ask whether a FinBERT model adapted to Indian financial text improves GJR-GARCH volatility forecasts. At the daily
level for TCS (2022–2026) it does not: once news volume is controlled for, headline sentiment adds nothing and performs
like shuffled noise. At the level of individual BSE announcements for six NSE-listed firms, it does. Scoring the
announcement's own text reduces the out-of-sample error of the reaction-day volatility forecast by 13.5 % on
results-related announcements (95 % CI −24.2 % to −1.8 %, n = 169, 2023–2026), relative to GARCH augmented with filing
type, company, company-specific results effects, release timing and recent volatility. The gain vanishes when each
announcement's scores are replaced by the same company's scores from another quarter (0/400 refits). Consistent with
the disclosure literature, text predicts the magnitude of the reaction, not its sign.

## 1 · Data

| | |
|---|---|
| Companies | TCS, Infosys, Wipro (IT) · Cipla, Dr Reddy's, Sun Pharma (pharma) |
| Announcements | 6,416 BSE corporate filings 2021-01 → 2026-06 with IST timestamps (SEBI LODR Reg 30 feed), collapsed to 4,231 firm-days |
| Text | filing headline + ~360-character summary of the filing |
| Prices | daily OHLC (NSE), dividend-adjusted |
| News (daily experiment) | ~310k Indian financial-news articles (Economic Times, Moneycontrol, Business Standard, Financial Express) |

**Realised variance** (daily, %²): `RV_t = 10⁴ [ ln(O_t / C_{t−1})² + ln(H_t / L_t)² / (4 ln 2) ]` — the overnight gap
(results are released after the close) plus the Parkinson intraday range.

## 2 · Models

**Language model.** FinBERT (ProsusAI) → domain-adaptive MLM pre-training on the Indian news corpus (FinBERT-IN) →
3-class sentiment head fine-tuned on target-aware inputs: a sentence pair (company name, headline with `<t>` around the
target, `<o>` around peers, tags on contrast words and market indices). Temperature-calibrated on a dev split.

**Volatility baseline.** GJR-GARCH(1,1,1) with Student-t errors on daily log returns, expanding window, re-estimated every
21 trading days on data strictly before each block: `h_{t+1|t}`.

**What text has to explain.** The GARCH miss on the reaction day R: `u_R = ln RV_R − ln h_{R|R−1}`. R is the
announcement's own trading day if released before 15:30 IST, otherwise the next trading day.

**Correction model.** `u_R = a + b′ x_meta + c′ x_text + ε`, OLS, expanding walk-forward by calendar year (fit on all
events before 1 Jan Y, predict Y = 2023…2026). Forecast `RV̂_R = h_R · exp(û_R + s²/2)`, s² from the training window.

| Set | Features |
|---|---|
| B0 | none (GARCH alone) |
| B1 metadata | filing type (9 groups), company, company × results-related, after-close flag, ln σ̂ firm, stock RV (1d, 5d), market EGARCH σ, ln # filings |
| B2 + FinBERT | B1 + s = P(pos) − P(neg), P(neg), P(pos), polarity 1 − P(neutral), max \|s\| that day, P(neg) × results |

## 3 · Results

**3.1 Announcement level** (out of sample 2023–2026; error = mean squared error of û; CIs: bootstrap over dates, 5,000 draws)

| Subset | n | GARCH | B1 | B2 | B2 vs B1 | 95 % CI |
|---|---|---|---|---|---|---|
| Results-related announcements | 169 | 1.857 | 1.493 | 1.291 | **−13.5 %** | [−24.2, −1.8] |
| · earnings-call transcripts | 66 | 1.095 | 1.321 | 1.028 | −22.2 % | [−41.3, −0.7] |
| · results releases | 85 | 2.641 | 1.704 | 1.492 | −12.4 % | [−25.3, +2.7] |
| · investor presentations | 18 | | 1.124 | 1.306 | +16.2 % | wide |
| All filings | 2,631 | 1.083 | 0.851 | 0.834 | −2.0 % | [−3.4, −0.6] |
| All except results-related | 2,462 | 1.030 | 0.807 | 0.803 | −0.5 % | [−1.0, +0.0] |
| Filed after the close | 1,856 | | 0.807 | 0.784 | −2.8 % | excludes 0 |
| Filed during trading | 775 | | 0.958 | 0.956 | −0.2 % | includes 0 |

Implied variance forecasts, all filings (QLIKE, lower is better): GARCH 0.591 · B1 0.531 · B2 0.520.
By year (all filings): 2023 −1.6 %, 2024 −1.7 %, 2025 −1.9 %, 2026 −3.5 % — same sign every year.

**3.2 Placebos** (B2 refitted with FinBERT columns permuted; 400 refits each; real B2 error 0.8344)

| Scheme | What it keeps | Mean placebo error | Refits ≤ real |
|---|---|---|---|
| P1 within filing type × year | how upbeat each kind of filing is in a given year | 0.8546 | 0 / 400 |
| P2 within filing type × company | the company's own writing style for that kind of filing | 0.8548 | 0 / 400 |

P2 rules out the main alternative explanation: that FinBERT merely identifies *which company* is reporting.

**3.3 What does not work**
- *Direction*: rank correlation of tone with the 2-day abnormal return ρ = +0.007 (p = 0.75).
- *Tail events*: probability of a > 2σ abnormal move — Brier 0.0584 (B1) vs 0.0585 (B2); AUC 0.740 vs 0.745; not significant.
- *Daily news sentiment* (TCS, test 2024 → Mar 2026): QLIKE GARCH 0.440 · + article counts 0.427 · + counts + FinBERT 0.433;
  sentiment vs counts p = 0.26; 78 % of within-year placebos do as well; the Model Confidence Set retains every model.
- *Domain adaptation for forecasting*: FinBERT-IN and off-the-shelf FinBERT give the same daily forecasts (p = 0.96), although
  FinBERT-IN classifies sentiment better (macro-F1 0.839 vs 0.757, provisional test labels).

## 3.4 · What exactly is being measured — event effect vs content effect

On a results day two different things raise volatility. The design separates them:

| Layer | What it captures | Error, results-related days | Change |
|---|---|---|---|
| GARCH | volatility expected from price history alone | 1.857 | — |
| + metadata = **event effect** | "an announcement of this type, from this company, at this time, came out" | 1.493 | −20 % |
| + FinBERT = **content effect** | "what *this* announcement says" | 1.291 | **−13.5 %** further |

The question is therefore not *does an announcement raise volatility?* (it does; the metadata layer absorbs that) but
*given that an announcement came out, does its text tell us how large the reaction will be?* The 13.5 % is the content
effect: the share of GARCH's reaction-day miss that the announcement's text predicts beyond the event itself.

Precise scope of the claim:

| Loose wording | What was actually done |
|---|---|
| "how much the stock moves" | how far reaction-day **volatility** (overnight gap² + Parkinson range) exceeds the GARCH forecast made the evening before — the *surprise* in volatility, not the raw move |
| "on results day" | the reaction day of **results-related announcements**: results releases (61), board-meeting outcomes (24), earnings-call transcripts (66), investor presentations (18). Transcripts are filed 1–8 days after the results, so many of the 169 days are not the results day itself |
| "sentiment of the earnings call" | FinBERT on the **filing headline + ~360-character summary**; for transcript filings this is a summary of the call, not the full transcript |

Claim this repository supports:

> For results-related BSE announcements (n = 169, out of sample 2023–2026), FinBERT scores of the announcement text
> reduce the reaction-day volatility forecast error by 13.5 % (95 % CI −24.2 % to −1.8 %) beyond GARCH and all
> metadata known at release time; the effect vanishes when the text is replaced by the same company's text from another
> quarter (0 / 400 refits).

**Open check — document type vs content.** Metadata treats the four results-related sub-types as one group. Since
transcript summaries read differently from results releases, part of the FinBERT effect could reflect *which kind of
document* it is rather than what it says; transcripts also arrive while post-results volatility is still decaying. The test:
add sub-type dummies, days-since-last-results-release, and sub-type × company to the metadata baseline, and re-run the
placebo within sub-type × company. If the effect survives, the content claim holds as stated; if it shrinks, the smaller
number is the one to report. Not yet run.

## 4 · Interpretation

Language carries information about *how much* uncertainty an event resolves, and it shows up where an event resolves a
lot of it: results-related disclosures, released after the close, read before the next session. Averaged over every trading
day the signal is diluted by quiet days and already captured by how much is written. Better sentiment classification does
not by itself mean better risk forecasts; the value appears only when the text is aligned with the event that moves the price.

## 5 · Limitations and next steps

- Six firms and 169 results-related days: the pooled effect is significant, individual sub-types are not all.
- Filing summaries rather than full documents; company-written text is rarely negative (10 / 6,416 filings).
- Specification history, for transparency: features and protocol were written and run on random placeholder scores
  before the real FinBERT scores existed. After the first real run (−17.5 % on results-related days), two stricter
  checks were added — the company × results-day control and the within-company placebo — which reduced the effect to
  the reported −13.5 %. Nothing was tuned on test-year performance.
- Next: all NIFTY 50 firms via the BSE fetcher (`python -m finbert_vol fetch-transcripts`, ~1,500 results days);
  a call-specific FinBERT for transcripts (largest effect, −22 %); options-implied volatility as a competing benchmark
  (`finbert_vol/options`).

## 6 · Claim → evidence

| Claim | File | Row / column |
|---|---|---|
| −13.5 % on results-related announcements | `results/experiments/event_reaction/t1_subsets.csv` | `results days`, `change_pct` |
| CI −24.2 % to −1.8 % | `experiments/verify_claims.py` output | recomputed from `t1_predictions.csv` |
| 0 / 400 placebos | `results/experiments/event_reaction/placebo.csv` | `t1_mse` ≥ 0.8344 in all rows |
| per-announcement forecasts | `results/experiments/event_reaction/t1_predictions.csv` | `u_react`, `u_hat_B0…B3`, `rv_hat_*` |
| daily null result | `results/experiments/finbert_vs_garch/summary.md` | contrasts, placebo, MCS |
| classifier macro-F1 | `results/sentiment/summary.md` | runs A_full, B, C |
