"""Build the domain-adaptation (MLM) corpus from the HF raw news files.

  python bert/prepare_mlm_corpus.py
    -> data/mlm/train.txt  (text dated < 2024-01-01, incl. older BS/FE archive)
       data/mlm/val.txt    (2024, sampled)        — perplexity / early stopping
       data/mlm/heldout_2025_2026.txt             — NEVER used in MLM (volatility test period)
       data/mlm/stats.json

One line per article: "<title>. <first ~600 chars of body, cut at a sentence end>".
Cleaning: HTML unescape, whitespace, drop podcasts / FE 'life' / empty bodies, dedupe on normalised title.
"""
from __future__ import annotations
import html, json, re
from pathlib import Path
import duckdb, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "hf" / "raw_dataset"; OUT = ROOT / "data" / "mlm"
SOURCES = ["businessstandard", "economictimes", "financialexpress", "moneycontrol"]
TRAIN_END, VAL_END = "2024-01-01", "2025-01-01"
BODY_CHARS, VAL_SAMPLE, SEED = 600, 20000, 42

def clean(t: str) -> str:
    t = html.unescape(t or "").replace("\xa0", " ")
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def lead(body: str, n: int = BODY_CHARS) -> str:
    b = clean(body)
    if len(b) <= n: return b
    cut = b[:n]; end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > n * 0.4 else cut

def main():
    con = duckdb.connect()
    u = " union all ".join(
        f"select '{s}' src, try_cast(date as date) d, title, news, url from read_csv('{RAW / (s + '_raw.csv')}', all_varchar=true)"
        for s in SOURCES)
    df = con.sql(f"""select * from ({u}) where d is not null and title is not null
                     and not regexp_matches(coalesce(url,''), '/(podcast|life|lifestyle|entertainment|sports)/')""").df()
    n0 = len(df)
    df["title"] = df["title"].map(clean)
    df["norm"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["norm"].str.len() > 15].sort_values("d").drop_duplicates("norm", keep="first")
    df["text"] = [f"{t.rstrip('.')}. {lead(b)}".strip() for t, b in zip(df["title"], df["news"].fillna(""))]
    df = df[df["text"].str.len() > 40]
    OUT.mkdir(parents=True, exist_ok=True)
    tr = df[df["d"] < pd.Timestamp(TRAIN_END)]
    va = df[(df["d"] >= pd.Timestamp(TRAIN_END)) & (df["d"] < pd.Timestamp(VAL_END))]
    va = va.sample(min(VAL_SAMPLE, len(va)), random_state=SEED)
    ho = df[df["d"] >= pd.Timestamp(VAL_END)]
    for name, part in [("train", tr.sample(frac=1, random_state=SEED)), ("val", va), ("heldout_2025_2026", ho)]:
        (OUT / f"{name}.txt").write_text("\n".join(part["text"].str.replace("\n", " ")) + "\n")
    stats = {"rows_raw": n0, "after_clean_dedupe": len(df), "train": len(tr), "val": len(va), "heldout": len(ho),
             "train_dates": [str(tr["d"].min().date()), str(tr["d"].max().date())],
             "approx_train_tokens_M": round(tr["text"].str.split().str.len().sum() * 1.3 / 1e6, 1),
             "train_by_source": tr["src"].value_counts().to_dict()}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2)); print(json.dumps(stats, indent=2))
    print("\nsample lines:"); [print("  ", l[:160]) for l in tr["text"].head(3)]

if __name__ == "__main__":
    main()
