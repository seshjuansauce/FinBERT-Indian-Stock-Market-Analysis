"""Score the HF news corpus for one target and build daily news features (row t = last trading day ≤ news date).

  python news/score_corpus.py                                  # attention only (no model needed)
  python news/score_corpus.py --model models/finbert-in-sentiment --tag C     # + FinBERT-IN sentiment
  python news/score_corpus.py --model ProsusAI/finbert --tag A                # + off-the-shelf FinBERT (full headline)

Outputs
  data/features/articles_tcs.csv            one row per relevant article (role, relevance, type, focus text, [sentiment_<tag>])
  data/features/news_daily_tcs[_<tag>].csv  one row per trading day t: attention, [sentiment, magnitude, carry, focus×carry]

Timing: the corpus has dates but no times. News dated d belongs to row t = the last trading day ≤ d, whose target is
RV_{t+1}. Weekend/holiday news therefore feeds the next session; nothing dated after t enters row t.
"""
from __future__ import annotations
import argparse, html, json, re, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "news")); sys.path.insert(0, str(ROOT / "model"))
from entity import analyse, model_input, focus_paragraph, FOCUS_TITLE
from news_features import carry

RAW, FEAT = ROOT / "data" / "hf" / "raw_dataset", ROOT / "data" / "features"
SOURCES = ["businessstandard", "economictimes", "financialexpress", "moneycontrol"]

def clean(t): return re.sub(r"\s+", " ", html.unescape(t or "")).strip()

def articles(target: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(ROOT / "news" / "config.yaml"))["companies"][target]
    u = " union all ".join(f"select '{s}' src, try_cast(date as date) d, title, news from read_csv('{RAW / (s + '_raw.csv')}', all_varchar=true)" for s in SOURCES)
    mm = cfg["must_match"].replace("'", "''")            # RE2 in DuckDB: filter before anything reaches pandas
    df = duckdb.sql(f"""select src, d, title, left(news, 4000) news from ({u})
                        where d >= '2020-01-01' and title is not null
                          and (regexp_matches(title, '{mm}') or regexp_matches(left(coalesce(news, ''), 400), '{mm}'))""").df()
    df["title"] = df["title"].map(clean); df["news"] = df["news"].fillna("").map(clean)
    if cfg.get("exclude"): df = df[~df["title"].str.contains(cfg["exclude"], regex=True)]
    df = df.copy()
    df["norm"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df["n_syndicated"] = df.groupby("norm")["src"].transform("nunique")
    df = df.sort_values("d").drop_duplicates("norm")          # first appearance of each headline
    rows = []
    for r in df.itertuples():
        a = analyse(r.title, r.news[:400], target)
        if a["relevance"] <= 0: continue
        is_focus = bool(FOCUS_TITLE.search(r.title))
        ftxt, fhow = focus_paragraph(r.news, target) if is_focus else ("", "")
        rows.append({"date": r.d, "source": r.src, "title": r.title, "role": a["role"], "relevance": a["relevance"],
                     "article_type": a["article_type"], "n_syndicated": r.n_syndicated, "is_focus": int(is_focus),
                     "focus_text": ftxt, "focus_how": fhow, "move_signal": a["move_signal"], "reco_signal": a["reco_signal"]})
    out = pd.DataFrame(rows); out["date"] = pd.to_datetime(out["date"])
    return out

def score(art: pd.DataFrame, model_path: str, target: str, bs: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Net sentiment P(pos) − P(neg) for each headline, and for each roundup's focus text (NaN if none)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path); mdl = AutoModelForSequenceClassification.from_pretrained(model_path).to(dev).eval()
    fmt_file = Path(model_path) / "input_format.json"
    fmt = json.loads(fmt_file.read_text()) if fmt_file.exists() else {"pair": False}
    lab = {i: l.lower() for i, l in mdl.config.id2label.items()}; ip = [i for i, l in lab.items() if l == "positive"][0]; ineg = [i for i, l in lab.items() if l == "negative"][0]
    def enc(text):
        if not fmt.get("pair"): return text, None
        return model_input(text, target, entity_markers=fmt["t"], other_markers=fmt["o"], contrast_tags=fmt["c"], market_tags=fmt["m"])
    @torch.no_grad()
    def run(texts):
        out = []
        for i in range(0, len(texts), bs):
            pairs = [enc(t) for t in texts[i:i + bs]]
            a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
            e = tok(a, b if fmt.get("pair") else None, truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
            pr = mdl(**e).logits.float().softmax(-1).cpu().numpy(); out.append(pr[:, ip] - pr[:, ineg])
        return np.concatenate(out) if out else np.zeros(0)
    s_title = run(art["title"].tolist())
    has = art["focus_text"].fillna("").str.len() > 40
    s_focus = np.full(len(art), np.nan); s_focus[has.values] = run(art.loc[has, "focus_text"].tolist())
    return s_title, s_focus

def daily(art: pd.DataFrame, trading_days: pd.DatetimeIndex, sent_col: str | None) -> pd.DataFrame:
    td = trading_days.sort_values()
    pos = td.searchsorted(art["date"].values, side="right") - 1          # last trading day ≤ news date
    t = pd.Series(td[np.clip(pos, 0, None)], index=art.index); t[pos < 0] = pd.NaT
    art = art.assign(t=t).dropna(subset=["t"])
    nf = art[art.is_focus == 0]; g = art.groupby("t"); gn = nf.groupby("t")
    d = pd.DataFrame(index=td)
    d["n_articles"] = g.size(); d["n_primary"] = art[art.role == "primary"].groupby("t").size()
    d["n_sources"] = g["source"].nunique(); d["n_focus"] = art[art.is_focus == 1].groupby("t").size()
    d["results_day"] = art[(art.article_type == "results") & (art.role == "primary")].groupby("t").size().gt(0).astype(float)
    d["syndication"] = g["n_syndicated"].max()
    d = d.fillna(0.0); d["log_n"] = np.log1p(d["n_articles"]); d["no_news"] = (d["n_articles"] == 0).astype(float)
    if sent_col:
        w = nf["relevance"]
        d["sent"] = (nf[sent_col] * w).groupby(nf["t"]).sum() / w.groupby(nf["t"]).sum()
        d["mag"] = (nf[sent_col].abs() * w).groupby(nf["t"]).sum() / w.groupby(nf["t"]).sum()
        d["sent_min"] = gn[sent_col].min()
        d["focus_para_sent"] = art.groupby("t")[sent_col + "_focus"].mean() if sent_col + "_focus" in art else np.nan
        # carry on the calendar-day series of non-roundup news (causal), sampled at each trading day
        cal = nf.assign(sw=nf[sent_col] * nf["relevance"]).groupby("date").agg(sw=("sw", "sum"), w=("relevance", "sum"))
        cal = pd.DataFrame({"sent": cal["sw"] / cal["w"], "w": cal["w"]})
        c = carry(cal).reindex(pd.date_range(min(cal.index.min(), td.min()), td.max()))
        d["carry"] = c.reindex(td).values
        d[["sent", "mag", "sent_min", "focus_para_sent", "carry"]] = d[["sent", "mag", "sent_min", "focus_para_sent", "carry"]].fillna(0.0)
        d["focus_x_carry"] = d["n_focus"] * d["carry"]
    d.index.name = "date"
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="TCS"); ap.add_argument("--model", default=None); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); tag = a.target.lower()
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / f"articles_{tag}.csv"
    art = pd.read_csv(path, parse_dates=["date"]) if path.exists() else articles(a.target)
    sent_col = None
    if a.model:
        t = a.tag or Path(a.model).name; sent_col = f"sent_{t}"
        art[sent_col], art[sent_col + "_focus"] = score(art, a.model, a.target)
    art.to_csv(path, index=False)
    px = pd.read_csv(FEAT / f"price_features_{tag}.csv", parse_dates=["date"])
    d = daily(art, pd.DatetimeIndex(px["date"]), sent_col)
    out = FEAT / (f"news_daily_{tag}.csv" if not sent_col else f"news_daily_{tag}_{sent_col[5:]}.csv")
    d.to_csv(out)
    print(f"articles {len(art)} ({art.date.min().date()} → {art.date.max().date()}), roles {art.role.value_counts().to_dict()}, "
          f"roundups {int(art.is_focus.sum())} (focus text {int((art.focus_how.fillna('') != '').sum()) if 'focus_how' in art else 0})")
    print(f"trading days with news {int((d.n_articles > 0).sum())}/{len(d)}, results days {int(d.results_day.sum())} → {out.name}")

if __name__ == "__main__":
    main()
