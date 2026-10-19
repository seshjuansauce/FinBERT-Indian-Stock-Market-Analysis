"""What do options say about volatility, compared with GARCH and the FinBERT model?

Inputs: the daily IV series (surface.constant_maturity) and a price table with rv, rv_target, gjr_fc, u_gjr and HAR
columns (data/features/price_features_<t>.csv + market RV, or the live state's LiveModel.table).

1. Level — variance risk premium (VRP). Options price volatility *risk*, so implied variance usually sits above what
   is later realised:
       σ²_imp,day = (100·σ_30)² / 252                         (%² per trading day)
       VRP_t      = σ²_imp,day,t − mean(RV_{t+1 … t+21})       (ex post, %²)       ratio = σ_imp / σ_realised
2. Information — Mincer–Zarnowitz regressions of next-day log RV on each forecast (Newey–West errors):
       ln RV_{t+1} = a + b · ln F_t + e        F ∈ {σ²_imp,day, h_GARCH}       b ≈ 1 and high R² = good forecast
   and an encompassing regression with both: whichever keeps its coefficient carries information the other lacks.
3. Forecasting — the same walk-forward correction of GARCH as the rest of the project, with option features added:
       u_{t+1} = a + b'·HAR_t + c'·OPT_t + e,   OPT = ln σ²_imp,day, skew, term slope        → QLIKE, Diebold–Mariano
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

HAR = ["lrv_d", "lrv_w", "lrv_m"]
OPT = ["liv", "skew_near", "term_slope"]


def join(iv: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    x = d.join(iv.set_index("date")[["iv30", "sigma_day_implied", "skew_near", "term_slope"]], how="left")
    x["iv_var_day"] = x["sigma_day_implied"] ** 2
    x["liv"] = np.log(x["iv_var_day"])
    x[["skew_near", "term_slope"]] = x[["skew_near", "term_slope"]].fillna(0.0)
    x["rv_fwd21"] = x["rv"][::-1].rolling(21, min_periods=15).mean()[::-1].shift(-1)       # mean RV over t+1 … t+21
    return x


def vrp(x: pd.DataFrame) -> dict:
    v = x.dropna(subset=["iv_var_day", "rv_fwd21"])
    if v.empty: return {}
    prem = v["iv_var_day"] - v["rv_fwd21"]
    return {"days": len(v), "implied_sigma_day_pct": round(float(np.sqrt(v["iv_var_day"]).mean()), 3),
            "realised_sigma_day_pct": round(float(np.sqrt(v["rv_fwd21"]).mean()), 3),
            "ratio_implied_to_realised": round(float(np.sqrt(v["iv_var_day"]).mean() / np.sqrt(v["rv_fwd21"]).mean()), 3),
            "vrp_mean_pct2": round(float(prem.mean()), 4), "share_days_vrp_positive": round(float((prem > 0).mean()), 3),
            "vrp_annualised_vol_points": round(float(np.sqrt(252 * max(v["iv_var_day"].mean(), 0)) - np.sqrt(252 * v["rv_fwd21"].mean())), 2)}


def mincer_zarnowitz(x: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    v = x.dropna(subset=["rv_target", "liv", "gjr_fc"]); y = np.log(v["rv_target"])
    rows = []
    for name, cols in {"implied (options)": ["liv"], "GJR-GARCH": ["lgarch"], "both": ["liv", "lgarch"]}.items():
        X = sm.add_constant(v.assign(lgarch=np.log(v["gjr_fc"]))[cols])
        r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"forecast": name, **{f"b_{c}": round(r.params[c], 3) for c in cols}, **{f"t_{c}": round(r.tvalues[c], 2) for c in cols},
                     "r2": round(r.rsquared, 3), "n": int(r.nobs)})
    return pd.DataFrame(rows)


def walk_forward(x: pd.DataFrame, feats: list[str] | None, start, refit=21, min_train=120) -> pd.Series:
    d = x.dropna(subset=["gjr_fc", "rv_target"] + HAR + (feats or []))
    if feats is None: return d["gjr_fc"]
    X, y = d[feats].values.astype(float), d["u_gjr"].values; out = pd.Series(np.nan, index=d.index)
    for s in range(max(d.index.searchsorted(pd.Timestamp(start)), min_train), len(d), refit):
        mu, sd = X[:s].mean(0), X[:s].std(0); sd[sd == 0] = 1
        Z = np.column_stack([np.ones(s), (X[:s] - mu) / sd]); b, *_ = np.linalg.lstsq(Z, y[:s], rcond=None)
        s2 = np.var(y[:s] - Z @ b, ddof=Z.shape[1]); e = min(s + refit, len(d))
        out.iloc[s:e] = d["gjr_fc"].values[s:e] * np.exp(np.column_stack([np.ones(e - s), (X[s:e] - mu) / sd]) @ b + s2 / 2)
    return out


def qlike(rv, f): z = np.asarray(rv) / np.asarray(f); return z - np.log(z) - 1


def dm(l1, l2, lag=5):
    dl = np.asarray(l1) - np.asarray(l2); n = len(dl); e = dl - dl.mean()
    v = e @ e / n + 2 * sum((1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, lag + 1))
    st = dl.mean() / math.sqrt(v / n) if v > 0 else 0.0
    return round(st, 2), round(math.erfc(abs(st) / math.sqrt(2)), 4)


def ladder(x: pd.DataFrame, extra: dict[str, list[str]] | None = None, start="2022-07-01", test_start="2024-01-01") -> pd.DataFrame:
    """QLIKE by period for GARCH, +HAR, +HAR+options (and any extra sets, e.g. news), on days that have IV."""
    sets = {"GJR-GARCH": None, "+HAR": HAR, "+HAR +options": HAR + OPT, **(extra or {})}
    have_iv = x["liv"].notna()
    fc = pd.DataFrame({k: walk_forward(x[have_iv], v, start) for k, v in sets.items()})
    fc["rv_target"] = x["rv_target"]; fc = fc.dropna()
    rows = []
    for per, m in {"validation": fc.index < pd.Timestamp(test_start), "test": fc.index >= pd.Timestamp(test_start)}.items():
        if m.sum() < 20: continue
        L = {k: qlike(fc.rv_target[m], fc[k][m]) for k in sets}
        for k in sets:
            st, p = dm(L[k], L["GJR-GARCH"]) if k != "GJR-GARCH" else (np.nan, np.nan)
            rows.append({"period": per, "model": k, "n": int(m.sum()), "qlike": round(float(L[k].mean()), 4),
                         "vs_garch_pct": round(100 * (L[k].mean() / L["GJR-GARCH"].mean() - 1), 1), "dm": st, "p": p})
    return pd.DataFrame(rows)
