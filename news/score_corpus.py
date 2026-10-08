"""Score the HF news corpus for one target and build daily news features (row t = last trading day ≤ news date).

  python news/score_corpus.py                                  # attention only (no model needed)
  python news/score_corpus.py --model models/finbert-in-sentiment --tag C     # + FinBERT-IN sentiment
  python news/score_corpus.py --model ProsusAI/finbert --tag A                # + off-the-shelf FinBERT (full headline)
  python news/score_corpus.py --tag C            # no model: re-derive sentiment from stored logits (CPU only), incl. the
                                                 # temperature-scaled version (tag Ct) if results/sentiment/calibration.json has C

Raw logits are stored per article (z_<tag>_<label>, zf_<tag>_<label> for roundup focus text), so calibration changes
never need the GPU again. Sentiment s = P(pos) − P(neg) with P = softmax(z / T).

Outputs
  data/features/articles_tcs.csv            one row per relevant article (role, relevance, type, focus text, [sentiment_<tag>])
  data/features/news_daily_tcs[_<tag>].csv  one row per trading day t: attention, [sentiment, magnitude, carry, focus×carry]

Timing: the corpus has dates but no times. News dated d belongs to row t = the last trading day ≤ d, whose target is
RV_{t+1}. Weekend/holiday news therefore feeds the next session; nothing dated after t enters row t.
"""
from __future__ import annotations
import argparse, html, json, re, sys, warnings
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

def score(art: pd.DataFrame, model_path: str, target: str, bs: int = 64) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Raw logits (n × 3) for each headline and for each roundup's focus text (NaN rows if none), plus label order."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path); mdl = AutoModelForSequenceClassification.from_pretrained(model_path).to(dev).eval()
    fmt_file = Path(model_path) / "input_format.json"
    fmt = json.loads(fmt_file.read_text()) if fmt_file.exists() else {"pair": False}
    labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
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
            out.append(mdl(**e).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(labels)))
    z_title = run(art["title"].tolist())
    has = art["focus_text"].fillna("").str.len() > 40
    z_focus = np.full((len(art), len(labels)), np.nan); z_focus[has.values] = run(art.loc[has, "focus_text"].tolist())
    return z_title, z_focus, labels

def sentiment_from_logits(art: pd.DataFrame, tag: str, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """s = P(pos) − P(neg), P = softmax(z / T), for headlines and focus texts."""
    def s(prefix):                                      # rows without focus text are all-NaN → NaN sentiment
        z = art[[f"{prefix}_{tag}_{l}" for l in ("positive", "negative", "neutral")]].to_numpy(float) / T
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = z - np.nanmax(z, 1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        return p[:, 0] - p[:, 1]
    return s("z"), s("zf")

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
    ap.add_argument("--temperature", default="auto", help="'auto' = results/sentiment/calibration.json[<calib-key>].T, or a number")
    ap.add_argument("--calib-key", default=None, help="model key in calibration.json (default: the tag)")
    a = ap.parse_args(); tg = a.target.lower()
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / f"articles_{tg}.csv"
    art = pd.read_csv(path, parse_dates=["date"]) if path.exists() else articles(a.target)
    tag = a.tag or (Path(a.model).name if a.model else None)
    if a.model:
        z, zf, labels = score(art, a.model, a.target)
        for j, l in enumerate(labels): art[f"z_{tag}_{l}"], art[f"zf_{tag}_{l}"] = z[:, j], zf[:, j]
    px = pd.read_csv(FEAT / f"price_features_{tg}.csv", parse_dates=["date"]); td = pd.DatetimeIndex(px["date"])
    outs = []
    if tag is None:
        d = daily(art, td, None); d.to_csv(FEAT / f"news_daily_{tg}.csv"); outs.append(f"news_daily_{tg}.csv")
    else:
        if f"z_{tag}_positive" not in art: raise SystemExit(f"no stored logits for tag {tag}: pass --model to score first")
        T = 1.0
        if a.temperature == "auto":
            cal = ROOT / "results" / "sentiment" / "calibration.json"
            key = a.calib_key or tag
            if cal.exists() and key in json.loads(cal.read_text()): T = float(json.loads(cal.read_text())[key]["T"])
        else:
            T = float(a.temperature)
        variants = [(tag, 1.0)] + ([(tag + "t", T)] if abs(T - 1.0) > 1e-6 else [])
        for vt, temp in variants:
            art[f"sent_{vt}"], art[f"sent_{vt}_focus"] = sentiment_from_logits(art, tag, temp)
            d = daily(art, td, f"sent_{vt}"); d.to_csv(FEAT / f"news_daily_{tg}_{vt}.csv")
            outs.append(f"news_daily_{tg}_{vt}.csv (T={temp:.2f}, share |s|>0.9 = {(art[f'sent_{vt}'].abs() > 0.9).mean():.2f})")
    art.to_csv(path, index=False)
    print(f"articles {len(art)} ({art.date.min().date()} → {art.date.max().date()}), roles {art.role.value_counts().to_dict()}, "
          f"roundups {int(art.is_focus.sum())} (focus text {int((art.focus_how.fillna('') != '').sum()) if 'focus_how' in art else 0})")
    print("wrote:", "; ".join(outs))

if __name__ == "__main__":
    main()
