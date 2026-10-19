"""Implied-volatility surface → a daily series of what the options market expects.

Per trading day t and expiry E (T = calendar days to expiry / 365):
  1. Forward F_E: the same-expiry futures close if it traded, else put–call parity F = K + e^{rT}(C − P) at the
     (up to) three strikes nearest the money where both a call and a put traded (median).
  2. Quotes: options that actually traded that day (volume > 0, close > 0), out of the money only
     (puts K < F, calls K ≥ F — the liquid side, and no early-exercise / intrinsic-value noise), |ln K/F| ≤ 0.25.
  3. σ_imp per quote from Black-76 (bs.implied_vol).
  4. ATM IV: linear interpolation of σ_imp in log-moneyness k = ln(K/F) at k = 0.
     Skew: σ(K = 0.95F) − σ(K = 1.05F)  (> 0: downside protection is dearer, the usual equity smile).
Per day:
  IV30: total variance w = σ²T interpolated linearly in T to 30 days between the two expiries that bracket it
        (flat-vol extrapolation if none do) → σ_30 = √(w30 / (30/365)).  The model-free 30-day expectation of vol.
  Implied daily σ (%):  σ_day = 100 · σ_30 / √252   (variance accrues on trading days only)
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

from .bs import implied_vols

K_BAND, MIN_DAYS = 0.25, 3


def forward(opts: pd.DataFrame, fut_close: float | None, T: float, r: float) -> tuple[float, str]:
    if fut_close and fut_close > 0: return float(fut_close), "futures"
    c = opts[opts.cp == "call"].set_index("strike")["close"]; p = opts[opts.cp == "put"].set_index("strike")["close"]
    both = c.index.intersection(p.index)
    if not len(both): return float("nan"), "none"
    diff = (c[both] - p[both]).abs().sort_values()                 # |C − P| smallest ≈ at the money
    Fs = [k + math.exp(r * T) * (c[k] - p[k]) for k in diff.index[:3]]
    return float(np.median(Fs)), "parity"


def _interp(k: np.ndarray, v: np.ndarray, at: float) -> float:
    o = np.argsort(k); k, v = k[o], v[o]
    if len(k) == 0 or at < k[0] - 1e-12 or at > k[-1] + 1e-12:
        return float(v[np.argmin(np.abs(k - at))]) if len(k) and abs(k[np.argmin(np.abs(k - at))] - at) < 0.02 else float("nan")
    return float(np.interp(at, k, v))


def expiry_slice(day: pd.DataFrame, expiry: pd.Timestamp, r: float) -> dict | None:
    t = day["date"].iloc[0]; T_days = (expiry - t).days
    if T_days < MIN_DAYS: return None
    T = T_days / 365.0
    opts = day[(day.kind == "option") & (day.expiry == expiry) & (day.volume > 0) & (day.close > 0)]
    fut = day[(day.kind == "future") & (day.expiry == expiry) & (day.volume > 0)]["close"]
    F, how = forward(opts, float(fut.iloc[0]) if len(fut) else None, T, r)
    if not np.isfinite(F) or opts.empty: return None
    k = np.log(opts["strike"].values / F)
    otm = ((opts.cp.values == "put") & (k < 0)) | ((opts.cp.values == "call") & (k >= 0))
    q = opts[otm & (np.abs(k) <= K_BAND)].copy()
    if len(q) < 2: return None
    q["k"] = np.log(q["strike"] / F); q["iv"] = implied_vols(q["close"].values, np.full(len(q), F), q["strike"].values, np.full(len(q), T), r, q["cp"].values)
    q = q[np.isfinite(q["iv"])]
    if len(q) < 2: return None
    kk, vv = q["k"].values, q["iv"].values
    atm = _interp(kk, vv, 0.0)
    return {"date": t, "expiry": expiry, "T_days": T_days, "F": F, "forward_from": how, "atm_iv": atm,
            "iv_95": _interp(kk, vv, math.log(0.95)), "iv_105": _interp(kk, vv, math.log(1.05)), "n_quotes": len(q),
            "underlying": float(day["underlying"].dropna().iloc[0]) if day["underlying"].notna().any() else float("nan")}


def surface(chain: pd.DataFrame, symbol: str, r: float = 0.065) -> pd.DataFrame:
    """One row per (date, expiry)."""
    c = chain[chain.symbol == symbol]; rows = []
    for _, day in c.groupby("date"):
        for e in sorted(day.loc[day.kind == "option", "expiry"].unique()):
            s = expiry_slice(day, pd.Timestamp(e), r)
            if s: rows.append(s)
    s = pd.DataFrame(rows)
    if len(s): s["skew"] = s["iv_95"] - s["iv_105"]
    return s


def constant_maturity(s: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Per date: σ_30 by total-variance interpolation, plus near/next ATM IV, term slope and near-expiry skew."""
    out = []
    for t, g in s.dropna(subset=["atm_iv"]).sort_values("T_days").groupby("date"):
        T, w = g["T_days"].values.astype(float), (g["atm_iv"].values ** 2) * g["T_days"].values / 365.0
        if len(T) == 1 or days <= T[0]: iv = g["atm_iv"].iloc[0]                       # flat extrapolation
        elif days >= T[-1]: iv = g["atm_iv"].iloc[-1]
        else:
            j = np.searchsorted(T, days); w30 = w[j - 1] + (w[j] - w[j - 1]) * (days - T[j - 1]) / (T[j] - T[j - 1])
            iv = math.sqrt(max(w30, 1e-12) / (days / 365.0))
        out.append({"date": t, f"iv{days}": iv, "atm_near": g["atm_iv"].iloc[0], "atm_next": g["atm_iv"].iloc[1] if len(g) > 1 else np.nan,
                    "near_days": int(T[0]), "skew_near": g["skew"].iloc[0], "F_near": g["F"].iloc[0], "n_expiries": len(g)})
    d = pd.DataFrame(out)
    if len(d):
        d["term_slope"] = d["atm_next"] - d["atm_near"]
        d["sigma_day_implied"] = 100 * d[f"iv{days}"] / math.sqrt(252)
    return d
