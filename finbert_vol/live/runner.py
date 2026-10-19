"""State + daily run. The state is a small folder (tar.gz in R2) — the container keeps nothing between runs.

state/
  prices.csv       Date, Symbol, Open, High, Low, Close, Adj Close, Volume, Dividends    (TCS.NS, ^CRSLDX)
  headlines.csv    published, date, title, source, url, role, relevance, article_type, s, p_*, model
  meta.json        news_through (last IST day fetched), model, created / updated
  public/          tracker.csv, latest.json, summary.json, headlines.json   (what the dashboard reads)
"""
from __future__ import annotations
import io
import json
import tarfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .gnews import GoogleNewsRSS, dedupe, norm_title
from .model import LiveModel

PRICE_COLS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends"]
HEAD_COLS = ["published", "date", "title", "source", "url", "role", "relevance", "article_type", "s", "p_positive", "p_negative",
             "p_neutral", "model"]


class YahooPrices:
    """Daily OHLCV from Yahoo Finance (yfinance), unadjusted + Adj Close, same layout as data/prices.csv."""

    def fetch(self, symbols: list[str], start: date) -> pd.DataFrame:
        import yfinance as yf
        out = []
        for s in symbols:
            h = yf.Ticker(s).history(start=str(start), auto_adjust=False, actions=True)
            if h.empty: raise RuntimeError(f"Yahoo returned no rows for {s} since {start}")
            h = h.reset_index().rename(columns={"index": "Date"})
            h["Date"] = pd.to_datetime(h["Date"]).dt.strftime("%Y-%m-%d"); h["Symbol"] = s
            if "Adj Close" not in h: h["Adj Close"] = h["Close"]
            if "Dividends" not in h: h["Dividends"] = 0.0
            out.append(h[PRICE_COLS])
        return pd.concat(out, ignore_index=True)


class State:
    def __init__(self, root: Path):
        self.root = Path(root); (self.root / "public").mkdir(parents=True, exist_ok=True)

    def read(self, name, **kw) -> pd.DataFrame | None:
        p = self.root / name
        return pd.read_csv(p, **kw) if p.exists() else None

    @property
    def meta(self) -> dict:
        p = self.root / "meta.json"; return json.loads(p.read_text()) if p.exists() else {}

    def save_meta(self, **kw):
        m = {**self.meta, **kw, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (self.root / "meta.json").write_text(json.dumps(m, indent=1))

    # tar.gz round trip (what the Worker stores in R2)
    @staticmethod
    def unpack(blob: bytes, root: Path) -> "State":
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        if blob:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for m in tf.getmembers():                 # refuse absolute paths / traversal
                    if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                        raise ValueError(f"unsafe path in state archive: {m.name}")
                tf.extractall(root)
        return State(root)

    def pack(self) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in sorted(self.root.rglob("*")):
                if p.is_file(): tf.add(p, arcname=str(p.relative_to(self.root)))
        return buf.getvalue()


def score_headlines(h: pd.DataFrame, model, target: str) -> pd.DataFrame:
    """Entity analysis (role / relevance / article type) + calibrated FinBERT sentiment for relevant headlines."""
    from entity import analyse
    if h.empty: return h.reindex(columns=HEAD_COLS)
    a = pd.DataFrame([analyse(t, "", target) for t in h["title"]], index=h.index)
    h = h.assign(role=a["role"], relevance=a["relevance"], article_type=a["article_type"])
    rel = h["relevance"] > 0
    for c in ("s", "p_positive", "p_negative", "p_neutral"): h[c] = np.nan
    if rel.any():
        pr = model.predict(h.loc[rel, "title"].tolist(), target)
        for c in ("s", "p_positive", "p_negative", "p_neutral"): h.loc[rel, c] = pr[c].values
    h["model"] = getattr(model, "name", "model")
    return h[HEAD_COLS]


class Runner:
    def __init__(self, state: State, sentiment_model, prices=None, news=None, target="TCS", symbol="TCS.NS", market="^CRSLDX",
                 today: date | None = None, log=print):
        self.st, self.model, self.target, self.symbol, self.market = state, sentiment_model, target, symbol, market
        self.prices_src = prices or YahooPrices(); self.news_src = news or GoogleNewsRSS(target)
        self.today = today or datetime.now(timezone(timedelta(hours=5, minutes=30))).date(); self.log = log

    # ---------- inputs ----------
    def update_prices(self) -> pd.DataFrame:
        px = self.st.read("prices.csv")
        start = (pd.to_datetime(px["Date"]).max() - pd.Timedelta(days=10)).date() if px is not None else date(2020, 1, 1)
        new = self.prices_src.fetch([self.symbol, self.market], start)
        px = pd.concat([px, new], ignore_index=True) if px is not None else new
        px = px.drop_duplicates(["Symbol", "Date"], keep="last").sort_values(["Symbol", "Date"])[PRICE_COLS]
        px.to_csv(self.st.root / "prices.csv", index=False)
        self.log(f"prices: {self.symbol} through {px[px.Symbol == self.symbol].Date.max()}")
        return px

    def update_news(self, start: date | None = None) -> pd.DataFrame:
        old = self.st.read("headlines.csv", parse_dates=["date"])
        through = self.st.meta.get("news_through")
        a = start or (date.fromisoformat(through) - timedelta(days=2) if through else self.today - timedelta(days=7))
        new = self.news_src.fetch(a, self.today, log=self.log)
        if old is not None and len(new):
            seen = set(zip(old["title"].map(norm_title), pd.to_datetime(old["date"])))
            new = new[[(k, d) not in seen for k, d in zip(new["title"].map(norm_title), pd.to_datetime(new["date"]))]]
        new = score_headlines(dedupe(new), self.model, self.target) if len(new) else new.reindex(columns=HEAD_COLS)
        h = pd.concat([old, new], ignore_index=True) if old is not None else new
        h = h.assign(published=pd.to_datetime(h["published"], utc=True, format="mixed")).sort_values("published")
        h.to_csv(self.st.root / "headlines.csv", index=False)
        self.st.save_meta(news_through=str(self.today), model=getattr(self.model, "name", "model"))
        self.log(f"news: +{len(new)} headlines ({int((new['relevance'] > 0).sum()) if len(new) else 0} relevant), {len(h)} total")
        return h

    # ---------- model + outputs ----------
    def publish(self) -> dict:
        px = self.st.read("prices.csv"); h = self.st.read("headlines.csv", parse_dates=["date"])
        lm = LiveModel(self.symbol, self.market)
        d = lm.table(px, h if h is not None else pd.DataFrame(columns=HEAD_COLS)); fc = lm.forecasts(d)
        pub = self.st.root / "public"
        lm.tracker(d, fc).to_csv(pub / "tracker.csv", index=False, float_format="%.5g")
        latest = lm.latest(d, fc); (pub / "latest.json").write_text(json.dumps(latest, indent=1))
        summary = {"metrics": lm.summary(fc, d), "coefficients_per_sd": lm.coefficients(d), "target": self.target,
                   "symbol": self.symbol, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "news_source": "Google News RSS: " + ", ".join(sorted(set(h["source"]))) if h is not None and len(h) else "none"}
        (pub / "summary.json").write_text(json.dumps(summary, indent=1))
        recent = (h[(h["relevance"] > 0) & (pd.to_datetime(h["date"]) >= pd.Timestamp(self.today) - pd.Timedelta(days=30))]
                  .sort_values("published", ascending=False).head(60) if h is not None else pd.DataFrame())
        recent = recent.assign(date=pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")) if len(recent) else recent
        (pub / "headlines.json").write_text(recent[["date", "title", "source", "url", "s", "role"]].round(3).to_json(orient="records")
                                            if len(recent) else "[]")
        self.log(f"forecast for {latest['forecast_for']}: σ̂ {latest['sigma_pct']}% (GARCH {latest['garch_sigma_pct']}%)")
        return latest

    def daily(self) -> dict:
        self.update_prices(); self.update_news(); return self.publish()
