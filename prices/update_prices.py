"""Stock prices from the axiom Postgres DB + yfinance delta to today -> data/prices.csv

Sources (only these two):
  1. yfinance: raw OHLC + Adj Close for the full range -> data/prices.csv (one consistent basis)
  2. axiom Postgres `fact_ohlcv` -> data/prices_db_adjusted.csv + cross-check. NOTE: the DB stores dividend-ADJUSTED closes,
     so it is not mixed into prices.csv (that was a bug in v1 of this script: double dividend adjustment).
     Docker stack must be up: docker compose -f ~/Desktop/Projects/axiom/infra/docker-compose.yml up -d

Output format (same as the HF nifty50_ticker.csv):
  Date,Symbol,Company,Index,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits
  Date like "2024-07-11 00:00:00+05:30"

  python prices/update_prices.py                 # default symbols (below)
  python prices/update_prices.py --all-db        # also every symbol present in the DB
  python prices/update_prices.py --start 2019-01-01

DB URL: AXIOM_DB_URL in .env, else axiom's default (postgresql+psycopg2://airflow:airflow@localhost:5432/axiom).
Cross-check printed per symbol: DB Close / yfinance Adj Close (should be a constant, sd ~0) and trading days missing in the DB.
"""
from __future__ import annotations
import argparse, os
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "data" / "prices.csv"
COLS = ["Date", "Symbol", "Company", "Index", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
SYMBOLS = {  # yf symbol -> (company, index label)
    "TCS.NS": ("Tata Consultancy Services Limited", "nifty50"),
    "INFY.NS": ("Infosys Limited", "nifty50"),
    "WIPRO.NS": ("Wipro Limited", "nifty50"),
    "HCLTECH.NS": ("HCL Technologies Limited", "nifty50"),
    "TECHM.NS": ("Tech Mahindra Limited", "nifty50"),
    "^NSEI": ("NIFTY 50", "index"),
    "^CNXIT": ("NIFTY IT", "index"),
    "^CRSLDX": ("NIFTY 500", "index"),
}
DEFAULT_DB = "postgresql+psycopg2://airflow:airflow@localhost:5432/axiom"

def env():
    p = ROOT / ".env"; e = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); e[k.strip()] = v.strip().strip('"').strip("'")
    return e

def from_db(symbols: list[str] | None, start: str) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    url = env().get("AXIOM_DB_URL") or os.environ.get("AXIOM_DB_URL") or DEFAULT_DB
    where = "AND i.yf_symbol = ANY(:syms)" if symbols else ""
    q = text(f"""SELECT i.yf_symbol AS "Symbol", i.company_name AS "Company", o.date AS "Date",
                        o.open AS "Open", o.high AS "High", o.low AS "Low", o.close AS "Close", o.volume AS "Volume",
                        o.dividends AS "Dividends", o.stock_splits AS "Stock Splits"
                 FROM fact_ohlcv o JOIN dim_instruments i ON i.id = o.instrument_id
                 WHERE o.interval = '1d' AND o.date >= :start {where} ORDER BY 1, 3""")
    try:
        with create_engine(url).connect() as c:
            df = pd.read_sql(q, c, params={"start": start, "syms": symbols} if symbols else {"start": start})
        print(f"DB: {len(df):,} rows, {df['Symbol'].nunique()} symbols")
        df["source"] = "db"; return df
    except Exception as exc:  # noqa: BLE001
        print(f"DB unavailable ({type(exc).__name__}: {str(exc)[:120]}) — using yfinance only. Is the docker stack up?")
        return pd.DataFrame()

def from_yf(sym: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    end = (date.today() + timedelta(days=1)).isoformat()
    h = yf.Ticker(sym).history(start=start, end=end, interval="1d", auto_adjust=False, actions=True)
    if h.empty: return pd.DataFrame()
    h = h.reset_index().rename(columns={"index": "Date"})
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
    h["Symbol"] = sym; h["source"] = "yfinance"
    return h[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "source"]]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2020-01-01"); ap.add_argument("--all-db", action="store_true")
    a = ap.parse_args()
    # The axiom DB stores DIVIDEND-ADJUSTED closes (verified: DB Close / yfinance Adj Close is constant per symbol).
    # So prices.csv is built on ONE basis: raw OHLC + Adj Close from yfinance for the full range; the DB is the cross-check.
    db = from_db(None if a.all_db else list(SYMBOLS), a.start)
    if len(db): db["Date"] = pd.to_datetime(db["Date"])
    syms = sorted(set(SYMBOLS) | (set(db["Symbol"]) if len(db) and a.all_db else set()))
    parts, report = [], []
    for s in syms:
        try:
            y = from_yf_full(s, a.start)
        except Exception as exc:  # noqa: BLE001
            print(f"  {s}: yfinance failed ({exc})"); y = pd.DataFrame()
        have = db[db["Symbol"] == s] if len(db) else pd.DataFrame()
        ratio_sd, ratio_med, n_overlap, missing_in_db = np.nan, np.nan, 0, np.nan
        if len(have) and len(y):
            m = have[["Date", "Close"]].merge(y[["Date", "Adj Close"]], on="Date")
            r = m["Close"].astype(float) / m["Adj Close"].astype(float)
            n_overlap, ratio_med, ratio_sd = len(m), float(r.median()), float(r.std())
            missing_in_db = len(set(y.loc[y["Date"].between(have["Date"].min(), have["Date"].max()), "Date"]) - set(have["Date"]))
        if len(y): parts.append(y)
        report.append((s, len(have), len(y), y["Date"].min().date() if len(y) else None, y["Date"].max().date() if len(y) else None,
                       n_overlap, ratio_med, ratio_sd, missing_in_db))
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df); df = df.dropna(subset=["Open", "High", "Low", "Close"])   # Yahoo returns today's bar incomplete until close
    if len(df) < n0: print(f"dropped {n0 - len(df)} incomplete rows (latest day not yet final) — re-run after market close")
    df["Company"] = df["Symbol"].map(lambda s: SYMBOLS.get(s, (s, None))[0])
    df["Index"] = df["Symbol"].map(lambda s: SYMBOLS.get(s, (None, "db"))[1])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d") + " 00:00:00+05:30"
    OUT.parent.mkdir(exist_ok=True); df[COLS].to_csv(OUT, index=False)
    if len(db):   # keep the raw DB export too, clearly labelled as adjusted
        d2 = db.copy(); d2["Date"] = d2["Date"].dt.strftime("%Y-%m-%d") + " 00:00:00+05:30"
        d2.drop(columns=["source"]).to_csv(OUT.with_name("prices_db_adjusted.csv"), index=False)
    print(f"\n{'symbol':11s} {'db_rows':>7s} {'yf_rows':>7s} {'yf_first':>10s} {'yf_last':>10s} {'overlap':>7s} {'DB/Adj med':>10s} {'sd':>8s} {'db_gaps':>7s}")
    for s, n, ny, f, l, no, rm, rs, mg in report:
        print(f"{s:11s} {n:7d} {ny:7d} {str(f):>10s} {str(l):>10s} {no:7d} {rm:10.4f} {rs:8.5f} {mg:7.0f}")
    print(f"\nwrote {len(df):,} rows → {OUT}  (yfinance raw OHLC + Adj Close)")
    print("Check: 'sd' should be ~0 — DB Close is yfinance Adj Close times a constant (dividends paid after the DB fetch).")

def from_yf_full(sym: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    end = (date.today() + timedelta(days=1)).isoformat()
    h = yf.Ticker(sym).history(start=start, end=end, interval="1d", auto_adjust=False, actions=True)
    if h.empty: return pd.DataFrame()
    h = h.reset_index().rename(columns={"index": "Date"})
    h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None).dt.normalize()
    h["Symbol"] = sym
    for c in ("Dividends", "Stock Splits"):
        if c not in h: h[c] = 0.0
    return h[["Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]

if __name__ == "__main__":
    main()
