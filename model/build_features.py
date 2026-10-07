"""Rung 0 — price-side features: realized-volatility target + walk-forward GARCH / GJR-GARCH forecasts.

  python model/build_features.py              # TCS.NS
  python model/build_features.py --check      # + verifies forecasts use no future data

Row = information date t (close of t). Everything in a row is known at t, except rv_target (= RV of t+1).
News dated t joins to row t.

Realized variance (%², daily):
  RV_t = 1e4 * [ ln(O_t / C_{t-1})^2  +  ln(H_t / L_t)^2 / (4 ln 2) ]
          overnight gap (results land after close)  +  Parkinson intraday range
  prices dividend-adjusted with factor AdjClose/Close, so ex-dividend gaps are not volatility.

GARCH(1,1) and GJR-GARCH(1,1,1), Student-t, on 100*log returns. Expanding window, refit every --refit days,
parameters estimated only on data strictly before each block. h_{t+1|t} = one-step variance forecast.
Residual target for the text model:  u_{t+1} = ln RV_{t+1} - ln h_{t+1|t}
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "features"
warnings.filterwarnings("ignore")

def load(symbol: str) -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data" / "prices.csv")
    p = p[p["Symbol"] == symbol].copy()
    p["date"] = pd.to_datetime(p["Date"].str[:10])
    p = p.sort_values("date").set_index("date")
    p = p[(p["Volume"] > 0) & (p["High"] > p["Low"])]   # yfinance emits flat zero-volume rows on some holidays
    f = p["Adj Close"] / p["Close"]
    for c in ["Open", "High", "Low", "Close"]:
        p[c] = p[c] * f
    return p[["Open", "High", "Low", "Close", "Volume", "Dividends"]]

def realized(p: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=p.index)
    d["ret"] = 100 * np.log(p["Close"] / p["Close"].shift(1))
    d["overnight"] = np.log(p["Open"] / p["Close"].shift(1))
    d["park_var"] = 1e4 * np.log(p["High"] / p["Low"]) ** 2 / (4 * np.log(2))
    d["rv"] = 1e4 * d["overnight"] ** 2 + d["park_var"]
    d["rv_target"] = d["rv"].shift(-1)
    # HAR inputs (log of daily / weekly / monthly average realized variance, known at t)
    d["lrv_d"] = np.log(d["rv"])
    d["lrv_w"] = np.log(d["rv"].rolling(5).mean())
    d["lrv_m"] = np.log(d["rv"].rolling(22).mean())
    d["ex_div"] = (p["Dividends"] > 0).astype(int)
    d["dow"] = d.index.dayofweek
    return d.iloc[1:]  # first row has no previous close

def walk_forward(r: pd.Series, o: int, burn: int, refit: int):
    """One-step variance forecasts h_{t+1|t} for t >= burn; params from data < block start."""
    fc, params = pd.Series(np.nan, index=r.index), []
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t")
    for s in range(burn, len(r), refit):
        res = am.fit(last_obs=s, disp="off")            # uses r[0:s] only
        e = min(s + refit, len(r))
        f = res.forecast(start=s, horizon=1, reindex=False).variance["h.1"]
        fc.iloc[s:e] = f.iloc[: e - s].values
        params.append({"block_start": str(r.index[s].date()), **res.params.round(5).to_dict()})
    return fc, params

def check_no_lookahead(r, params, fc, o, n=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(np.where(fc.notna())[0], n, replace=False)
    starts = pd.to_datetime([b["block_start"] for b in params])
    for t in idx:
        b = params[np.searchsorted(starts, r.index[t], side="right") - 1]
        pv = pd.Series({k: v for k, v in b.items() if k != "block_start"})
        m = arch_model(r.iloc[: t + 1], mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t")
        h = m.fix(pv).forecast(horizon=1, reindex=False).variance.iloc[-1, 0]   # data up to t only
        assert abs(h - fc.iloc[t]) / fc.iloc[t] < 2e-3, (r.index[t], h, fc.iloc[t])
    print(f"look-ahead check passed on {n} random origins (refit on truncated data reproduces forecast)")

def qlike(rv, h): x = rv / h; return float(np.mean(x - np.log(x) - 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="TCS.NS"); ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--refit", type=int, default=21); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = realized(load(a.symbol)); r = d["ret"]
    meta = {"symbol": a.symbol, "burn": a.burn, "refit": a.refit, "params": {}}
    for name, o in [("garch", 0), ("gjr", 1)]:
        fc, params = walk_forward(r, o, a.burn, a.refit)
        d[f"{name}_fc"] = fc
        d[f"u_{name}"] = np.log(d["rv_target"]) - np.log(fc)
        meta["params"][name] = params
        if a.check: check_no_lookahead(r, params, fc, o)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.symbol.split(".")[0].lower()
    d.to_csv(OUT / f"price_features_{tag}.csv", index_label="date")
    (OUT / f"garch_params_{tag}.json").write_text(json.dumps(meta, indent=1))

    ev = d.dropna(subset=["garch_fc", "gjr_fc", "rv_target"])
    print(f"\n{a.symbol}: {len(d)} days {d.index[0].date()} → {d.index[-1].date()}; "
          f"out-of-sample {len(ev)} days from {ev.index[0].date()}")
    print(f"RV (daily %²): median {d['rv'].median():.2f}, mean {d['rv'].mean():.2f}, "
          f"overnight share of RV {np.mean(1e4*d['overnight']**2 / d['rv']):.0%}")
    print(f"\n{'model':8s} {'QLIKE':>7s} {'RMSE lnRV':>10s} {'MZ R²':>7s} {'MZ b':>6s}   mean u")
    for m in ["garch", "gjr"]:
        y, x = np.log(ev["rv_target"]), np.log(ev[f"{m}_fc"])
        b, c = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f"{m:8s} {qlike(ev['rv_target'], ev[f'{m}_fc']):7.3f} {np.sqrt(np.mean((y-x)**2)):10.3f} "
              f"{r2:7.3f} {b:6.2f}   {ev[f'u_{m}'].mean():+.3f}")
    last = meta["params"]["gjr"][-1]
    print(f"\nlast GJR params ({last['block_start']}): " + ", ".join(f"{k}={v}" for k, v in last.items() if k != "block_start"))
    big = ev["u_gjr"].nlargest(8)
    print("\nlargest GJR misses (u = ln RV_{t+1} − ln h): info date → target day")
    for t, u in big.items():
        nxt = d.index[d.index.get_loc(t) + 1].date()
        print(f"  {t.date()} → {nxt}  u={u:+.2f}  RV={ev.loc[t,'rv_target']:.1f}  h={ev.loc[t,'gjr_fc']:.2f}")
    print(f"\nwrote {OUT / f'price_features_{tag}.csv'}")

if __name__ == "__main__":
    main()
