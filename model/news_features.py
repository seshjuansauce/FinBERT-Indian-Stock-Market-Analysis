"""Daily news features joined to the price rows (row t = information known at close of t).

carry_t  — decayed sentiment of the target's recent news, used to give "stocks in focus" roundups
           the sentiment of the catalyst that put the stock in focus:

    carry_t = Σ_{d ∈ (t−W, t]} w_{t−d} s_d / Σ w_{t−d},   w_k = 2^(−k/h),  W = 30 days, h = 5 days

    s_d = relevance-weighted entity-targeted sentiment of the target's NON-roundup articles on calendar day d.
    Only days ≤ t enter, so no look-ahead. Days without articles contribute nothing (not zeros).

focus features:
    n_focus_t          number of roundups listing the target on day t
    focus_x_carry_t    n_focus_t × carry_t       (being in focus amplifies the recent story)
    focus_para_sent_t  sentiment of the target's own paragraph inside the roundup (when extractable)
"""
from __future__ import annotations
import numpy as np, pandas as pd

def carry(daily: pd.DataFrame, window: int = 30, half_life: float = 5.0) -> pd.Series:
    """daily: index = calendar dates (DatetimeIndex), columns 'sent' (mean sentiment that day) and 'w' (article weight).
    Returns carry on every calendar day in the span (NaN where no article in the window)."""
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    s = daily["sent"].reindex(days).fillna(0.0).to_numpy()
    w = daily["w"].reindex(days).fillna(0.0).to_numpy()
    k = np.arange(window)                               # lags 0 … W-1  → days (t−W, t]
    decay = 2.0 ** (-k / half_life)
    num = np.convolve(s * w, decay)[: len(days)]          # causal: only lags ≥ 0
    den = np.convolve(w, decay)[: len(days)]
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=days, name="carry")

def focus_features(daily_all: pd.DataFrame, daily_nonfocus: pd.DataFrame, window=30, half_life=5.0) -> pd.DataFrame:
    """daily_all: per-day n_focus (+ optional focus_para_sent); daily_nonfocus: per-day sent, w of non-roundup news."""
    c = carry(daily_nonfocus, window, half_life)
    out = pd.DataFrame(index=c.index).join(c)
    out["n_focus"] = daily_all["n_focus"].reindex(out.index).fillna(0)
    if "focus_para_sent" in daily_all: out["focus_para_sent"] = daily_all["focus_para_sent"].reindex(out.index)
    out["focus_x_carry"] = out["n_focus"] * out["carry"].fillna(0)
    return out

if __name__ == "__main__":   # self-test
    idx = pd.to_datetime(["2024-01-01", "2024-01-06", "2024-02-20"])
    d = pd.DataFrame({"sent": [1.0, -1.0, 0.5], "w": [1.0, 1.0, 1.0]}, index=idx)
    c = carry(d)
    assert c["2024-01-01"] == 1.0                                   # only itself
    assert abs(c["2024-01-06"] - (2 ** -1 * 1 - 1) / (2 ** -1 + 1)) < 1e-12   # 5-day-old +1 weighs half of today's −1
    assert np.isnan(c["2024-02-10"])                                # > 30 days after last article → no carry
    assert c["2024-01-05"] == 1.0                                   # before the −1 arrives: no look-ahead
    f = focus_features(pd.DataFrame({"n_focus": [2]}, index=pd.to_datetime(["2024-01-06"])), d)
    assert abs(f.loc["2024-01-06", "focus_x_carry"] - 2 * c["2024-01-06"]) < 1e-12
    print("carry self-test passed:", c[["2024-01-01", "2024-01-05", "2024-01-06", "2024-01-30"]].round(3).to_dict())
