"""Sample TCS headlines for sentiment labelling (train pool 2020–2023, blind test 2024).

  python bert/label_sample.py  ->  data/labels/sample_train.csv (400), data/labels/sample_test.csv (150)

Only articles where TCS is a real subject (entity.py relevance > 0). Train is stratified to over-represent
TCS-primary headlines and headlines with move / reco / analyst cues; test is a plain random draw from 2024
(representative, so test accuracy reflects real data). 2025–26 is never touched (volatility test period).
"""
from __future__ import annotations
import hashlib, html, re, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "news"))
from entity import analyse

RAW, OUT = ROOT / "data" / "hf" / "raw_dataset", ROOT / "data" / "labels"
SEED, N_TRAIN, N_TEST = 42, 400, 150

def clean(t): return re.sub(r"\s+", " ", html.unescape(t or "")).strip()

def pool():
    cfg = yaml.safe_load(open(ROOT / "news" / "config.yaml"))["companies"]["TCS"]
    u = " union all ".join(f"select '{s}' src, try_cast(date as date) d, title, news from "
                           f"read_csv('{RAW / (s + '_raw.csv')}', all_varchar=true)"
                           for s in ["businessstandard", "economictimes", "financialexpress", "moneycontrol"])
    df = duckdb.sql(f"select * from ({u}) where d between '2020-01-01' and '2024-12-31' and title is not null").df()
    df["title"] = df["title"].map(clean); df["news"] = df["news"].map(clean)
    m = df["title"].str.contains(cfg["must_match"], regex=True) & ~df["title"].str.contains(cfg["exclude"], regex=True)
    df = df[m].copy()
    df["norm"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df = df.sort_values("d").drop_duplicates("norm")
    rows = []
    for r in df.itertuples():
        a = analyse(r.title, r.news[:300], "TCS")
        if a["relevance"] <= 0: continue
        rows.append({"id": hashlib.md5(r.title.encode()).hexdigest()[:10], "date": str(r.d)[:10], "source": r.src,
                     "title": r.title, "snippet": r.news[:280], "target_span": a["target_span"], "role": a["role"],
                     "article_type": a["article_type"], "cue": int(bool(a["move_signal"] or a["reco_signal"] or a["analyst_signal"]))})
    return pd.DataFrame(rows)

def main():
    p = pool(); p["year"] = p["date"].str[:4].astype(int)
    print("relevant TCS headlines:", len(p)); print(p.groupby(["year", "role"]).size().unstack(fill_value=0))
    tr_pool, te_pool = p[p.year <= 2023], p[p.year == 2024]
    rng = np.random.default_rng(SEED)
    # train: weight primary x2 and cue x2, sample without replacement
    w = (1 + (tr_pool["role"] == "primary")) * (1 + tr_pool["cue"]); w = w / w.sum()
    tr = tr_pool.iloc[rng.choice(len(tr_pool), min(N_TRAIN, len(tr_pool)), replace=False, p=w.values)]
    te = te_pool.sample(min(N_TEST, len(te_pool)), random_state=SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["id", "date", "source", "title", "snippet", "target_span", "role", "article_type", "cue"]
    tr.sort_values("date")[cols].to_csv(OUT / "sample_train.csv", index=False)
    te.sort_values("date")[cols].to_csv(OUT / "sample_test.csv", index=False)
    for n, s in [("train", tr), ("test", te)]:
        print(f"\n{n}: {len(s)}  roles {s.role.value_counts().to_dict()}  cue {s.cue.mean():.0%}  types {s.article_type.value_counts().head(5).to_dict()}")

if __name__ == "__main__":
    main()
