"""Live GARCH–FinBERT model: one table, three walk-forward forecasts, a tracker for the dashboard.

    ln RV_{t+1} = ln h_{t+1|t} + u_{t+1}                       h: GJR-GARCH(1,1,1)-t, refit every 21 days
    u_{t+1}     = a + b'·HAR_t + c'·NEWS_t + e                 OLS, walk-forward (each block fits on rows before it)
    σ̂²_{t+1}   = h_{t+1|t} · exp(â + b̂'HAR_t + ĉ'NEWS_t + s²/2)

    HAR_t  = ln RV_t, ln RV̄_{t−4..t}, ln RV̄_{t−21..t}, ln RV_mkt,t, ln RV̄_mkt,t−4..t    (market = NIFTY 500)
    NEWS_t = log(1 + n_t), 1[n_t = 0], s̄_t, |s|̄_t                n_t = relevant headlines mapped to row t
                                                               s = P(pos) − P(neg), FinBERT-IN, temperature-calibrated

Forecasts:  garch (h only) · har (+HAR) · finbert (+HAR +NEWS).  Timing: a headline dated d (IST) belongs to the last
trading day ≤ d, so evening news after the close of t counts for the forecast of t+1.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("model", "news"):
    if str(ROOT / sub) not in sys.path: sys.path.insert(0, str(ROOT / sub))

HAR = ["lrv_d", "lrv_w", "lrv_m", "lmkt_d", "lmkt_w"]
NEWS = ["log_n", "no_news", "sent", "mag"]
SETS = {"garch": None, "har": HAR, "finbert": HAR + NEWS}


def adjusted(px: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """OHLC of one symbol, dividend-adjusted with AdjClose/Close (same as model/build_features.load)."""
    p = px[px["Symbol"] == symbol].copy()
    p["date"] = pd.to_datetime(p["Date"].astype(str).str[:10]); p = p.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    p = p[(p["Volume"].fillna(0) > 0) | symbol.startswith("^")]; p = p[p["High"] > p["Low"]]
    f = (p["Adj Close"] / p["Close"]).fillna(1.0)
    for c in ["Open", "High", "Low", "Close"]: p[c] = p[c] * f
    p["Dividends"] = p.get("Dividends", 0.0); p["Dividends"] = p["Dividends"].fillna(0.0)
    return p[["Open", "High", "Low", "Close", "Volume", "Dividends"]]


def market_rv(p: pd.DataFrame) -> pd.DataFrame:
    rv = 1e4 * (np.log(p.Open / p.Close.shift(1)) ** 2 + np.log(p.High / p.Low) ** 2 / (4 * np.log(2)))
    return pd.DataFrame({"lmkt_d": np.log(rv), "lmkt_w": np.log(rv.rolling(5).mean())})


def news_daily(h: pd.DataFrame, td: pd.DatetimeIndex) -> pd.DataFrame:
    """Relevant headlines → per-trading-day features (row = last trading day ≤ headline date)."""
    d = pd.DataFrame(index=td)
    h = h[h["relevance"] > 0] if len(h) else h
    if len(h):
        pos = td.searchsorted(pd.to_datetime(h["date"]).values, side="right") - 1
        h = h.assign(t=np.where(pos >= 0, td[np.clip(pos, 0, None)], pd.NaT)).dropna(subset=["t"])
        g = h.groupby("t"); w = h["relevance"]
        d["n"] = g.size()
        d["sent"] = (h["s"] * w).groupby(h["t"]).sum() / w.groupby(h["t"]).sum()
        d["mag"] = (h["s"].abs() * w).groupby(h["t"]).sum() / w.groupby(h["t"]).sum()
    for c in ("n", "sent", "mag"):
        if c not in d: d[c] = 0.0
    d = d.fillna(0.0); d["log_n"] = np.log1p(d["n"]); d["no_news"] = (d["n"] == 0).astype(float)
    return d


class LiveModel:
    def __init__(self, symbol="TCS.NS", market="^CRSLDX", burn=500, refit=21, min_train=120, eval_start="2022-07-01",
                 test_start="2024-01-01"):
        self.symbol, self.market, self.burn, self.refit, self.min_train = symbol, market, burn, refit, min_train
        self.eval_start, self.test_start = pd.Timestamp(eval_start), pd.Timestamp(test_start)

    # ---------- table ----------
    def table(self, prices: pd.DataFrame, headlines: pd.DataFrame) -> pd.DataFrame:
        from build_features import realized, walk_forward
        p = adjusted(prices, self.symbol); d = realized(p); d["close"] = p["Close"].reindex(d.index)
        fc, _ = walk_forward(d["ret"], 1, self.burn, self.refit)              # GJR: o=1
        d["gjr_fc"] = fc; d["u_gjr"] = np.log(d["rv_target"]) - np.log(d["gjr_fc"])
        d = d.join(market_rv(adjusted(prices, self.market)), how="left")
        d[["lmkt_d", "lmkt_w"]] = d[["lmkt_d", "lmkt_w"]].ffill()
        return d.join(news_daily(headlines, d.index)[NEWS + ["n"]], how="left").fillna({c: 0.0 for c in NEWS + ["n"]})

    # ---------- forecasts ----------
    @staticmethod
    def _ols(y, X):
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return beta, mu, sd, float(np.var(y - Z @ beta, ddof=Z.shape[1]))

    def forecasts(self, d: pd.DataFrame) -> pd.DataFrame:
        """σ̂² for t+1 on every row from eval_start (incl. the last row, whose target is still unknown)."""
        d = d.dropna(subset=["gjr_fc"] + HAR)
        out = pd.DataFrame(index=d.index); out["rv_target"] = d["rv_target"]
        known = d["rv_target"].notna().values
        i0 = d.index.searchsorted(self.eval_start)
        for name, feats in SETS.items():
            if feats is None: out[name] = d["gjr_fc"]; continue
            X = d[feats].values.astype(float); y = d["u_gjr"].values; f = pd.Series(np.nan, index=d.index)
            for s in range(max(i0, self.min_train), len(d), self.refit):
                tr = np.arange(s)[known[:s]]
                beta, mu, sd, s2 = self._ols(y[tr], X[tr])
                e = min(s + self.refit, len(d)); Z = np.column_stack([np.ones(e - s), (X[s:e] - mu) / sd])
                f.iloc[s:e] = d["gjr_fc"].values[s:e] * np.exp(Z @ beta + s2 / 2)
            out[name] = f
        return out.loc[out.index >= self.eval_start]

    def coefficients(self, d: pd.DataFrame) -> dict:
        """Full-sample fit of the finbert set (for display): coefficient per 1 sd of each feature."""
        d = d.dropna(subset=["gjr_fc", "rv_target"] + HAR); feats = SETS["finbert"]
        beta, *_ = self._ols(d["u_gjr"].values, d[feats].values.astype(float))
        return {f: round(float(b), 4) for f, b in zip(feats, beta[1:])}

    # ---------- evaluation ----------
    @staticmethod
    def qlike(rv, f): x = np.asarray(rv) / np.asarray(f); return x - np.log(x) - 1

    @staticmethod
    def dm(l1, l2, lag=5) -> tuple[float, float]:
        """Diebold–Mariano on loss differential l1 − l2, Newey–West variance. Negative stat: model 1 better."""
        dlt = np.asarray(l1) - np.asarray(l2); n = len(dlt); m = dlt.mean(); e = dlt - m
        v = e @ e / n + 2 * sum((1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, lag + 1))
        st = m / math.sqrt(v / n) if v > 0 else 0.0
        return float(st), float(math.erfc(abs(st) / math.sqrt(2)))

    def summary(self, fc: pd.DataFrame, d: pd.DataFrame) -> dict:
        f = fc.dropna(); res = {}
        for per, m in {"validation": f.index < self.test_start, "test": f.index >= self.test_start, "all": f.index >= f.index.min()}.items():
            if m.sum() < 10: continue
            L = {k: self.qlike(f.rv_target[m], f[k][m]) for k in SETS}
            st, p = self.dm(L["finbert"], L["garch"])
            res[per] = {"n": int(m.sum()), "qlike": {k: round(float(v.mean()), 4) for k, v in L.items()},
                        "finbert_vs_garch_pct": round(100 * (L["finbert"].mean() / L["garch"].mean() - 1), 1),
                        "dm_stat": round(st, 2), "dm_p": round(p, 4),
                        # share of days with |next-day return| inside ±σ̂ (≈ 0.68 if σ̂ is right and returns were normal)
                        "hit_rate": round(float(np.mean(np.abs(d["ret"].shift(-1).reindex(f.index[m]).values) <= np.sqrt(f["finbert"][m].values))), 3)}
        return res

    # ---------- outputs ----------
    def tracker(self, d: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
        """One row per *target* day: the forecast made the evening before, and what happened."""
        idx = d.index; nxt = pd.Series(list(idx[1:]) + [pd.NaT], index=idx)
        t = fc.index; target = nxt.reindex(t)
        if pd.isna(target.iloc[-1]): target.iloc[-1] = t[-1] + pd.offsets.BDay(1)          # pending: next weekday
        out = pd.DataFrame({"day": target.values, "pred_vol": np.sqrt(fc["finbert"].values), "garch_vol": np.sqrt(fc["garch"].values),
                            "har_vol": np.sqrt(fc["har"].values), "actual_vol": np.sqrt(fc["rv_target"].values),
                            "news_n": d.loc[t, "n"].values, "news_sent": d.loc[t, "sent"].values})
        px = d[["close", "ret"]].shift(0)
        out["close"] = px["close"].reindex(out["day"]).values; out["ret"] = px["ret"].reindex(out["day"]).values
        out["period"] = np.where(pd.to_datetime(out["day"]) >= self.test_start, "test", "validation")
        out["pending"] = out["actual_vol"].isna()
        return out[out["pred_vol"].notna()].reset_index(drop=True)

    def latest(self, d: pd.DataFrame, fc: pd.DataFrame) -> dict:
        last = fc.index[-1]
        return {"as_of": str(last.date()), "forecast_for": str((last + pd.offsets.BDay(1)).date()),
                "sigma_pct": round(float(np.sqrt(fc["finbert"].iloc[-1])), 3),
                "garch_sigma_pct": round(float(np.sqrt(fc["garch"].iloc[-1])), 3),
                "news_today": int(d.loc[last, "n"]), "news_sent_today": round(float(d.loc[last, "sent"]), 3),
                "close": round(float(d.loc[last, "close"]), 2)}
