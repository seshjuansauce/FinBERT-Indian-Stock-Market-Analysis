# Labelling rubric: TCS-targeted headline sentiment

Label each headline by asking: **"Reading only this headline (snippet for context), is the news good, bad or neither for TCS (Tata Consultancy Services) and its share price?"**
Judge TCS only — not the market, not other companies in the headline.

positive — clearly good for TCS: results beat / profit or revenue growth highlighted, large deal win or contract, stock rose/gained/jumped, upgrade / target raised / buy recommendation, buyback, guidance or demand commentary upbeat, market-share or valuation milestone.
negative — clearly bad for TCS: results miss / profit or revenue fall / margin pressure, stock fell/slipped/dragged index, downgrade / target cut / sell, client loss or deal cancellation, penalties, lawsuits with financial stakes, layoffs framed as weakness, attrition spike, weak guidance or demand warnings.
neutral — no clear direction for TCS: appointments, product or partnership launches without financial magnitude, event/date announcements (e.g. "TCS to announce Q2 results on Oct 9"), explainers, general lists ("stocks to watch: TCS, Infosys…") with no TCS-specific direction, CSR/brand/award news, genuinely mixed news.

Rules
1. Lists / co-mentions: use only the part about TCS. "Infosys, TCS drag Sensex" → negative. "Sensex falls; TCS gains" → positive. "Stocks to watch: TCS, Wipro, RIL" → neutral.
2. Mixed results ("profit up, revenue misses estimates"): follow the headline's emphasis; if balanced → neutral with low confidence.
3. Questions / previews ("What to expect from TCS Q3?") → neutral unless the headline itself states a direction ("TCS Q3 preview: weak quarter likely" → negative).
4. Dividends alone → neutral (routine); a special dividend or buyback → positive.
5. Don't use outside knowledge of what the stock later did.
6. Acquisitions by TCS → positive (inorganic growth; carve-outs of a client's IT unit usually come with a multi-year
   services contract). Negative only if the headline itself flags overpaying, dilution or investor concern.
7. The label is always for the **marked target** (`<t>`), which is TCS for TCS headlines and the peer for peer headlines.

## Contrast rules (two clauses pulling in opposite directions)
| Structure | Rule | Example → label |
|---|---|---|
| Concessive: despite, in spite of, even as, even though, although, though, notwithstanding | **Main clause decides**, wherever it sits; the despite-clause is the concession | "TCS slips 2% even as results beat" → negative; "Despite Q1 miss, brokerages bullish on TCS" → positive |
| Adversative: but, however, ", yet" | **Clause after the connective decides** (if it is still about the target) | "TCS posts strong quarter, but guidance cut" → negative |
| Adversative that switches company | Judge only the target's clause | "TCS shares fall but Infosys defiant" → negative |
| No connective, two facts | Headline emphasis = first clause | "TCS beats estimates, profit declines 3.2%" → positive |
| Not contrast | "but for" = except; temporal "yet" ("no slowdown yet"); "amid" = circumstance | label normally |


Output: a CSV with header `id,label,confidence,reason` — label ∈ {positive, negative, neutral}; confidence ∈ {high, medium, low}; reason ≤ 18 words, quoting the decisive words. One row per input row, same ids, no omissions. Quote the reason field.
