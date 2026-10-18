"""Earnings-call sentences for labelling (FinBERT-IN-call).

  python bert/call_label_sample.py
    data/calls/transcripts.jsonl          the IT transcripts (persisted copy; source: axiom BSE filings)
    data/calls/call_sentences.csv         every parsed sentence: call_id, company, published, role, section, context, flags
    data/labels/calls_sample_train.csv    500 sentences from calls published ≤ 2023 (stratified)
    data/labels/calls_sample_test.csv     150 sentences from calls published ≥ 2024 (random) — label blind
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from finbert_vol.calls import TranscriptParser
from finbert_vol.config import ISIN, Settings

SEED, N_TRAIN, N_TEST = 42, 500, 150

def main():
    s = Settings.load(); axiom = Path(s.axiom_dir).expanduser()
    f = pd.read_parquet(axiom / "research" / "data" / "filings.parquet")
    src = Path(os.environ.get("SRC_JSONL", Path.home() / "transcripts_full.jsonl"))
    T = {}
    for p in [src, ROOT / "data/calls/transcripts.jsonl"]:
        if p.exists():
            for l in open(p): j = json.loads(l); T[j["p"]] = j["t"]
    rows, keep = [], []
    parser = TranscriptParser()
    for t in s.call_targets:
        tr = f[(f.bse_subcategory == "Earnings Call Transcript") & (f["isin"] == ISIN[t])].sort_values("date")
        for r in tr.itertuples():
            if r.path not in T: continue
            keep.append({"p": r.path, "t": T[r.path]})
            pc = parser.parse(T[r.path])
            if len(pc.sentences) < 50: continue                       # garbled extraction
            cid = f"{t}_{pd.Timestamp(r.date):%Y%m%d}"
            for i, x in enumerate(pc.sentences.itertuples()):
                rows.append({"sid": f"{cid}_{i:04d}", "call_id": cid, "company": t, "published": str(pd.Timestamp(r.date).date()),
                             "call_time": str(pc.call_time) if pc.call_time is not None else "", "speaker": x.speaker, "role": x.role,
                             "section": x.section, "context": x.context, "sentence": x.sentence,
                             "hedge": int(x.hedge), "forward": int(x.forward), "number": int(x.number)})
    out = ROOT / "data/calls"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "transcripts.jsonl", "w") as fh:
        for k in keep: fh.write(json.dumps(k) + "\n")
    S = pd.DataFrame(rows); S.to_csv(out / "call_sentences.csv", index=False)
    print(f"{S.call_id.nunique()} calls, {len(S)} sentences; by company {S.groupby('company').call_id.nunique().to_dict()}")
    rng = np.random.default_rng(SEED)
    tr, te = S[S.published < "2024-01-01"], S[S.published >= "2024-01-01"]
    # train: equal thirds across prepared / answers / questions, and over-weight hedged and forward-looking sentences
    w = (1 + tr.hedge + tr.forward).astype(float)
    parts = []
    for ctx, g in tr.groupby("context"):
        n = N_TRAIN // 3 + (N_TRAIN % 3 if ctx == "management answer" else 0)
        parts.append(g.iloc[rng.choice(len(g), min(n, len(g)), replace=False, p=(w[g.index] / w[g.index].sum()).values)])
    trs = pd.concat(parts).sample(frac=1, random_state=SEED)
    tes = te.sample(min(N_TEST, len(te)), random_state=SEED)
    cols = ["sid", "call_id", "company", "published", "role", "context", "sentence", "hedge", "forward", "number"]
    lab = ROOT / "data/labels"; lab.mkdir(parents=True, exist_ok=True)
    trs[cols].to_csv(lab / "calls_sample_train.csv", index=False); tes[cols].to_csv(lab / "calls_sample_test.csv", index=False)
    for n, d in [("train", trs), ("test", tes)]:
        print(f"{n}: {len(d)}  calls {d.call_id.nunique()}  context {d.context.value_counts().to_dict()}  hedge {d.hedge.mean():.0%} forward {d.forward.mean():.0%}")

if __name__ == "__main__":
    main()
