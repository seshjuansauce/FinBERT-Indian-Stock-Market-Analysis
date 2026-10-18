# Labelling rubric: earnings-call sentence sentiment

Each row is one sentence from an Indian IT company's earnings call (TCS, Infosys or Wipro — column `company`).
`context` says who is speaking and where: `management prepared` (scripted opening remarks), `management answer`
(answering an analyst in Q&A), `analyst question` (a sell-side analyst asking).

Ask: **"What does this sentence convey about the company's business performance or outlook?"**

positive — growth, strong/healthy demand, deal wins, margin expansion, beating plans, confidence, an upbeat outlook;
           an analyst praising results ("congrats on a strong quarter").
negative — decline, weakness, slowdown, delays, cuts, pricing/margin pressure, client stress, risks that are materialising,
           a cautious or pessimistic outlook; an analyst question that **presupposes a problem** ("Why did BFSI slow so sharply?",
           "Are you worried margins will fall further?").
neutral  — procedural lines (introductions, safe-harbour boilerplate, "next question please"), facts with no direction
           (definitions, structure, headcount split), neutral information requests ("Can you share the deal-win split by
           geography?"), or genuinely balanced statements.

Rules
1. Hedging alone is not negative. "We may see some improvement next quarter" is positive (low confidence);
   "We may see some softness" is negative. Judge the content, not the word "may".
2. Contrast: in "Y despite X" / "Y even as X" the main clause Y decides; in "X, but Y" (also "yet", "however", "whereas", "while" used as contrast) the clause after the connective decides.
3. Numbers: follow the stated direction (grew 3% → positive; declined → negative; flat → neutral unless framed as good/bad).
4. Forward-looking statements: label the expected direction ("we expect the pipeline to stay robust" → positive).
5. Analyst questions: label the tone the question carries or presupposes, not the answer you imagine.
6. Thank-yous and pleasantries that slipped through → neutral.
7. Don't use outside knowledge of what the stock did.

Output CSV header `sid,label,confidence,reason` — label ∈ {positive, negative, neutral}; confidence ∈ {high, medium, low};
reason ≤ 18 words quoting the decisive words. One row per input sid, no omissions, properly quoted.
