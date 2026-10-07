"""Entity-aware FinBERT scoring of news, then daily features per company.

  python news/score_news.py
    -> data/news_scored.csv : one row per (article, company): role, article_type, relevance, target_span,
                              FinBERT on the target span (fb_t_*) AND on the full normalised title (fb_f_*)
    -> data/news_daily.csv  : one row per company × day (attention + relevance-weighted sentiment)
    -> prints accuracy vs data/gold_headlines.csv (if present): full-title vs target-span FinBERT

Articles at/after 15:30 IST count toward the NEXT day (market closed).
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "news"))
from entity import analyse   # noqa: E402
MODEL = "ProsusAI/finbert"

def finbert(texts: list[str], batch: int = 32) -> pd.DataFrame:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL); mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"; mdl.to(dev)
    labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    uniq = list(dict.fromkeys(texts)); out = []
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            enc = tok(uniq[i:i + batch], truncation=True, max_length=96, padding=True, return_tensors="pt").to(dev)
            out.append(torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy())
    P = pd.DataFrame(np.vstack(out), columns=labels, index=uniq)
    return P.loc[texts].reset_index(drop=True)

def label_of(df: pd.DataFrame, pre: str) -> pd.Series:
    return df[[f"{pre}positive", f"{pre}negative", f"{pre}neutral"]].idxmax(axis=1).str.replace(pre, "", regex=False)

def main():
    raw = pd.read_csv(DATA / "news_raw.csv")
    raw = raw.drop_duplicates(["article_id", "company"]).reset_index(drop=True)
    feats = pd.DataFrame([analyse(t, d if isinstance(d, str) else "", c) for t, d, c in zip(raw["title"], raw["description"], raw["company"])])
    df = pd.concat([raw, feats], axis=1)
    # near-duplicate titles across outlets (after normalisation) -> keep first, count the copies as syndication
    df["dup_count"] = df.groupby(["company", "title_norm"])["article_id"].transform("count")
    df = df.drop_duplicates(["company", "title_norm"]).reset_index(drop=True)
    spans = df["target_span"].where(df["target_span"].str.len() > 0, df["title_norm"]).tolist()
    T = finbert(spans).add_prefix("fb_t_"); F = finbert(df["title_norm"].tolist()).add_prefix("fb_f_")
    df = pd.concat([df, T, F], axis=1)
    for p in ("fb_t_", "fb_f_"):
        df[p + "net"] = df[p + "positive"] - df[p + "negative"]; df[p + "label"] = label_of(df, p)
    ts = pd.to_datetime(df["pubDate"], utc=True).dt.tz_convert("Asia/Kolkata")
    after = (ts.dt.hour * 60 + ts.dt.minute) >= 15 * 60 + 30
    df["pub_ist"] = ts
    df["day"] = (ts.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after.astype(int), unit="D")).dt.date
    df.to_csv(DATA / "news_scored.csv", index=False)

    # adjusted target sentiment: an explicit price-move word overrides FinBERT (it misses market slang)
    df["sent_adj"] = np.where(df["move_signal"] != 0, df["move_signal"].astype(float), df["fb_t_net"])
    df["adj_label"] = np.select([df["sent_adj"] > 0.3, df["sent_adj"] < -0.3], ["positive", "negative"], "neutral")
    rows = []
    for (c, d), g in df.groupby(["company", "day"]):
        rel = g[g["relevance"] > 0]; w = rel["relevance"]
        focal = rel[rel["role"].isin(["primary", "co_mention", "sector"])]       # min/max only over company-focused items
        wavg = lambda col: float((w * rel[col]).sum() / w.sum()) if w.sum() else np.nan
        rows.append(dict(company=c, day=d,
            # attention
            n_articles=len(g), n_syndicated=int(g["dup_count"].sum()), n_primary=int((g["role"] == "primary").sum()),
            n_sources=g["source_id"].nunique(),
            n_results_news=int(g["article_type"].isin(["results", "results_preview", "results_date"]).sum()),
            n_market_wrap=int(g["article_type"].isin(["market_wrap", "live_updates", "stocks_to_watch"]).sum()),
            # entity sentiment (relevance-weighted)
            sent_target_w=wavg("fb_t_net"), sent_adj_w=wavg("sent_adj"),
            sent_target_min=float(focal["fb_t_net"].min()) if len(focal) else np.nan,
            sent_adj_min=float(focal["sent_adj"].min()) if len(focal) else np.nan,
            sent_primary=float(g.loc[g["role"] == "primary", "sent_adj"].mean()) if (g["role"] == "primary").any() else np.nan,
            # explicit signals (counts; not "sentiment")
            move_net=int(rel["move_signal"].sum()), n_move_neg=int((rel["move_signal"] < 0).sum()), n_move_pos=int((rel["move_signal"] > 0).sum()),
            reco_net=int(g["reco_signal"].sum()), n_buy=int((g["reco_signal"] > 0).sum()), n_sell=int((g["reco_signal"] < 0).sum()),
            analyst_net=int(g["analyst_signal"].sum()),
            # naive baseline kept only for comparison
            sent_full_mean=float(g["fb_f_net"].mean()),
        ))
    daily = pd.DataFrame(rows); daily.to_csv(DATA / "news_daily.csv", index=False)
    print(f"scored {len(df):,} (article, company) rows → news_scored.csv | {len(daily):,} company-days → news_daily.csv")

    gold_p = DATA / "gold_headlines.csv"
    if gold_p.exists():
        gold = pd.read_csv(gold_p)
        gold = gold[gold["label_target"].isin(["positive", "negative", "neutral"])]
        m = gold.merge(df[["article_id", "company", "fb_t_label", "fb_f_label", "adj_label", "role", "target_span"]], on=["article_id", "company"])
        if len(m):
            acc = lambda col, x=m: (x[col] == x["label_target"]).mean()
            nn = m[m["label_target"] != "neutral"]
            flips = lambda col, x=m: int(((x[col] == "positive") & (x["label_target"] == "negative")).sum() + ((x[col] == "negative") & (x["label_target"] == "positive")).sum())
            print(f"\nvs gold ({len(m)} headlines, {int(len(nn))} non-neutral; labeler={gold['labeler'].iloc[0]})")
            print(f"  {'method':22s} {'acc':>5s} {'acc non-neutral':>16s} {'sign flips':>11s}")
            for col, name in [("fb_f_label", "FinBERT full headline"), ("fb_t_label", "FinBERT target span"), ("adj_label", "target + move words")]:
                print(f"  {name:22s} {acc(col):5.2f} {acc(col, nn):16.2f} {flips(col):11d}")
            print(m[m["adj_label"] != m["label_target"]][["company", "label_target", "adj_label", "target_span"]].to_string(index=False))

if __name__ == "__main__":
    main()
