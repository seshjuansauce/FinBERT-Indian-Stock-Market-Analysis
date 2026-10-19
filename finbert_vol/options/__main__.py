"""python -m finbert_vol.options <command>      (run on your Mac — NSE blocks most cloud IPs)

  fetch    [--symbols TCS,INFY] [--start 2022-01-01] [--end today]   daily F&O bhavcopies → data/options/raw/*.parquet
  analyse  [--symbol TCS] [--r 0.065]                                surface → IV30 series → VRP, Mincer–Zarnowitz, QLIKE ladder
                                                                     → results/options/{surface,iv_daily}_<sym>.csv, report_<sym>.json
  price    --F 4000 --K 4100 --T-days 30 --sigma 0.22 [--r 0.065]   Black-76 price + Greeks of one call and put (a calculator)
  iv       --price 52.3 --F 4000 --K 4100 --T-days 30 --cp call      implied vol of one quote
"""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(prog="finbert_vol.options")
    ap.add_argument("cmd", choices=["fetch", "analyse", "price", "iv"])
    ap.add_argument("--symbols", default="TCS"); ap.add_argument("--symbol", default="TCS")
    ap.add_argument("--start", default="2022-01-01"); ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--r", type=float, default=0.065, help="risk-free rate, continuous (India 91-day T-bill ≈ 6–7%%)")
    ap.add_argument("--F", type=float); ap.add_argument("--K", type=float); ap.add_argument("--T-days", type=float)
    ap.add_argument("--sigma", type=float); ap.add_argument("--price", type=float); ap.add_argument("--cp", default="call")
    a = ap.parse_args()

    if a.cmd in ("price", "iv"):
        from .bs import greeks, implied_vol
        T = a.T_days / 365
        if a.cmd == "iv":
            print(f"implied vol = {implied_vol(a.price, a.F, a.K, T, a.r, a.cp):.4f}"); return
        for kind in ("call", "put"):
            g = greeks(a.F, a.K, T, a.r, a.sigma, kind)
            print(f"{kind:4s} price {float(g['price']):9.3f}  Δ {float(g['delta']):+.4f}  Γ {float(g['gamma']):.6f}  "
                  f"vega/vol-pt {float(g['vega']) / 100:.3f}  Θ/day {float(g['theta']) / 365:+.3f}")
        return

    from .data import NSEBhavcopy
    cache = ROOT / "data" / "options" / "raw"
    if a.cmd == "fetch":
        syms = a.symbols.split(",")
        df = NSEBhavcopy(cache, syms).range(date.fromisoformat(a.start), date.fromisoformat(a.end))
        print(f"{len(df):,} rows, {df['date'].nunique() if len(df) else 0} days, symbols {sorted(df['symbol'].unique()) if len(df) else []}")
        return

    from . import compare
    from .surface import constant_maturity, surface
    files = sorted(cache.glob("*.parquet"))
    if not files: raise SystemExit("no bhavcopies cached — run `python -m finbert_vol.options fetch` first")
    chain = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    s = surface(chain, a.symbol, a.r); iv = constant_maturity(s)
    out = ROOT / "results" / "options"; out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / f"surface_{a.symbol.lower()}.csv", index=False); iv.to_csv(out / f"iv_daily_{a.symbol.lower()}.csv", index=False)
    print(f"{a.symbol}: {len(s)} expiry-days, {len(iv)} days with IV30 ({iv.date.min():%Y-%m-%d} → {iv.date.max():%Y-%m-%d}); "
          f"forwards from futures {100 * (s.forward_from == 'futures').mean():.0f}%")

    px = pd.read_csv(ROOT / "data" / "features" / f"price_features_{a.symbol.lower()}.csv", parse_dates=["date"]).set_index("date")
    x = compare.join(iv, px)
    rep = {"symbol": a.symbol, "r": a.r, "vrp": compare.vrp(x), "mincer_zarnowitz": compare.mincer_zarnowitz(x).to_dict("records")}
    news = ROOT / "data" / "features" / f"news_daily_{a.symbol.lower()}.csv"
    extra = {}
    if news.exists():
        nd = pd.read_csv(news, parse_dates=["date"]).set_index("date"); x = x.join(nd[["log_n"]], how="left"); x["log_n"] = x["log_n"].fillna(0)
        extra = {"+HAR +news": compare.HAR + ["log_n"], "+HAR +options +news": compare.HAR + compare.OPT + ["log_n"]}
    lad = compare.ladder(x, extra); rep["ladder"] = lad.to_dict("records")
    (out / f"report_{a.symbol.lower()}.json").write_text(json.dumps(rep, indent=1, default=float))
    print("\nVariance risk premium:", rep["vrp"])
    print("\nMincer–Zarnowitz  ln RV_{t+1} = a + b·ln F_t:\n", compare.mincer_zarnowitz(x).to_string(index=False))
    print("\nWalk-forward QLIKE (lower is better):\n", lad.to_string(index=False))


if __name__ == "__main__":
    main()
