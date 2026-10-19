"""Options module: Black–Scholes/Black-76 maths, NSE bhavcopy parsing (both formats), IV surface, comparison."""
import math

import numpy as np
import pandas as pd
import pytest

from finbert_vol.options.bs import black76, bounds, bsm, forward_from_parity, greeks, implied_vol
from finbert_vol.options.data import normalise, url_for
from finbert_vol.options.surface import constant_maturity, surface


# ---------------- Black–Scholes ----------------
def test_hull_textbook_example():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, σ=20%, T=0.5 → c = 4.76, p = 0.81
    assert float(bsm(42, 40, 0.5, 0.10, 0.20, kind="call")) == pytest.approx(4.76, abs=0.005)
    assert float(bsm(42, 40, 0.5, 0.10, 0.20, kind="put")) == pytest.approx(0.81, abs=0.005)


def test_put_call_parity_and_forward_recovery():
    F, K, T, r, s = 4000.0, 4150.0, 40 / 365, 0.065, 0.24
    c, p = float(black76(F, K, T, r, s, "call")), float(black76(F, K, T, r, s, "put"))
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), rel=1e-10)
    assert forward_from_parity(c, p, K, T, r) == pytest.approx(F, rel=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [3500.0, 4000.0, 4600.0])
def test_implied_vol_round_trip(kind, K):
    F, T, r = 4000.0, 25 / 365, 0.065
    for s in (0.08, 0.22, 0.6):
        price = float(black76(F, K, T, r, s, kind)); iv = implied_vol(price, F, K, T, r, kind)
        assert float(black76(F, K, T, r, iv, kind)) == pytest.approx(price, abs=1e-6)   # reprices to within ₹0.000001
        # deep in the money with little time value, σ is poorly identified (vega ≈ 0) — why the surface uses OTM quotes
        assert iv == pytest.approx(s, abs=1e-6 if abs(math.log(K / F)) < 0.1 else 1e-4)


def test_implied_vol_outside_bounds_is_nan():
    F, K, T, r = 4000.0, 3900.0, 30 / 365, 0.065
    lb, ub = bounds(F, K, T, r, "call")
    assert math.isnan(implied_vol(lb * 0.99, F, K, T, r, "call")) and math.isnan(implied_vol(ub * 1.01, F, K, T, r, "call"))
    assert math.isnan(implied_vol(10.0, F, K, 0.0, r, "call"))


def test_greeks_match_finite_differences():
    F, K, T, r, s, h = 4000.0, 4100.0, 30 / 365, 0.065, 0.25, 1e-4
    for kind in ("call", "put"):
        g = greeks(F, K, T, r, s, kind); P = lambda f=F, v=s, t=T: float(black76(f, K, t, r, v, kind))
        assert float(g["delta"]) == pytest.approx((P(F + h * F) - P(F - h * F)) / (2 * h * F), rel=1e-4)
        assert float(g["gamma"]) == pytest.approx((P(F + h * F) - 2 * P() + P(F - h * F)) / (h * F) ** 2, rel=1e-3)
        assert float(g["vega"]) == pytest.approx((P(v=s + h) - P(v=s - h)) / (2 * h), rel=1e-4)
        assert float(g["theta"]) == pytest.approx(-(P(t=T + h) - P(t=T - h)) / (2 * h), rel=1e-3)   # ∂/∂(calendar time) = −∂/∂T


# ---------------- NSE data ----------------
def test_url_switches_format_on_udiff_date():
    from datetime import date
    assert url_for(date(2024, 7, 5)).endswith("/DERIVATIVES/2024/JUL/fo05JUL2024bhav.csv.zip")
    assert url_for(date(2024, 7, 8)).endswith("/fo/BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip")


def test_normalise_old_and_udiff():
    old = pd.DataFrame({"INSTRUMENT": ["OPTSTK", "FUTSTK", "OPTIDX"], "SYMBOL": ["TCS", "TCS", "NIFTY"], "EXPIRY_DT": ["25-Jul-2024"] * 3,
                        "STRIKE_PR": [4000, 0, 24000], "OPTION_TYP": ["CE", "XX", "PE"], "OPEN": 1, "HIGH": 1, "LOW": 1,
                        "CLOSE": [80.5, 4010.0, 100], "SETTLE_PR": [80.5, 4010.0, 100], "CONTRACTS": [120, 3000, 5], "VAL_INLAKH": 1,
                        "OPEN_INT": [5000, 90000, 1], "CHG_IN_OI": 0, "TIMESTAMP": ["04-JUL-2024"] * 3})
    n = normalise(old)
    assert n.loc[0, ["kind", "cp", "strike", "close"]].tolist() == ["option", "call", 4000, 80.5]
    assert n.loc[1, "kind"] == "future" and np.isnan(n.loc[1, "strike"]) and n.loc[0, "date"] == pd.Timestamp("2024-07-04")
    new = pd.DataFrame({"TradDt": ["2024-07-08"] * 2, "BizDt": "x", "Sgmt": "FO", "Src": "NSE", "FinInstrmTp": ["STO", "STF"],
                        "FinInstrmId": 1, "ISIN": "", "TckrSymb": ["TCS", "TCS"], "SctySrs": "", "XpryDt": ["2024-07-25"] * 2,
                        "FininstrmActlXpryDt": "", "StrkPric": [3900.0, np.nan], "OptnTp": ["PE", np.nan], "FinInstrmNm": "",
                        "OpnPric": 1, "HghPric": 1, "LwPric": 1, "ClsPric": [42.0, 4020.0], "LastPric": 1, "PrvsClsgPric": 1,
                        "UndrlygPric": [4005.0, 4005.0], "SttlmPric": [42.0, 4020.0], "OpnIntrst": [100, 200], "ChngInOpnIntrst": 0,
                        "TtlTradgVol": [10, 20], "TtlTrfVal": 1, "TtlNbOfTxsExctd": 1, "SsnId": "F1", "NewBrdLotQty": 175})
    m = normalise(new)
    assert m.loc[0, ["kind", "cp", "strike", "underlying"]].tolist() == ["option", "put", 3900.0, 4005.0]
    assert m.loc[1, "kind"] == "future" and m.loc[1, "cp"] == ""


# ---------------- surface ----------------
def _chain(day, F, T_days_list, smile, r=0.065, futures=True):
    rows = []
    for Td in T_days_list:
        e = pd.Timestamp(day) + pd.Timedelta(days=Td); T = Td / 365
        for K in np.arange(0.8, 1.21, 0.025) * F:
            for cp in ("call", "put"):
                s = smile(math.log(K / F), Td)
                rows.append({"date": pd.Timestamp(day), "symbol": "TCS", "kind": "option", "expiry": e, "strike": round(K, 2), "cp": cp,
                             "close": float(black76(F, K, T, r, s, cp)), "settle": np.nan, "underlying": np.nan, "oi": 10, "volume": 5})
        if futures:
            rows.append({"date": pd.Timestamp(day), "symbol": "TCS", "kind": "future", "expiry": e, "strike": np.nan, "cp": "",
                         "close": F, "settle": F, "underlying": np.nan, "oi": 10, "volume": 5})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("futures", [True, False])
def test_surface_recovers_atm_skew_and_iv30(futures):
    smile = lambda k, Td: 0.22 - 0.30 * k + (0.02 if Td > 40 else 0.0)        # downside skew; next month 2 vol points higher
    ch = _chain("2025-03-03", 4000.0, [24, 52], smile, futures=futures)
    s = surface(ch, "TCS")
    near = s.sort_values("T_days").iloc[0]
    assert near["F"] == pytest.approx(4000.0, rel=1e-6) and near["forward_from"] == ("futures" if futures else "parity")
    assert near["atm_iv"] == pytest.approx(0.22, abs=2e-3)
    assert near["skew"] == pytest.approx(0.30 * (math.log(1.05) - math.log(0.95)), abs=3e-3)
    d = constant_maturity(s).iloc[0]
    w = lambda iv, Td: iv ** 2 * Td / 365
    expect = math.sqrt((w(0.22, 24) + (w(0.24, 52) - w(0.22, 24)) * (30 - 24) / (52 - 24)) / (30 / 365))
    assert d["iv30"] == pytest.approx(expect, abs=2e-3) and d["term_slope"] == pytest.approx(0.02, abs=2e-3)
    assert d["sigma_day_implied"] == pytest.approx(100 * d["iv30"] / math.sqrt(252))


def test_surface_skips_untraded_and_expiring():
    ch = _chain("2025-03-03", 4000.0, [2, 24], lambda k, Td: 0.25)
    ch.loc[(ch.kind == "option") & (ch.strike > 4000), "volume"] = 0          # no traded OTM calls in either expiry
    s = surface(ch, "TCS")
    assert s["T_days"].tolist() == [24]                                        # 2-day expiry dropped
    assert s.iloc[0]["atm_iv"] == pytest.approx(0.25, abs=3e-3)                # still found from puts at/below the money


# ---------------- comparison ----------------
def test_compare_on_synthetic_data():
    from finbert_vol.options import compare
    rng = np.random.default_rng(0); n = 700; idx = pd.bdate_range("2022-01-03", periods=n)
    lv = np.log(2.0) + np.cumsum(rng.normal(0, 0.05, n)) * 0.3                  # latent log variance (%², daily)
    rv = np.exp(lv + rng.normal(0, 0.6, n))
    d = pd.DataFrame({"rv": rv}, index=idx); d["rv_target"] = d["rv"].shift(-1)
    for c, w in (("lrv_d", 1), ("lrv_w", 5), ("lrv_m", 22)): d[c] = np.log(d["rv"].rolling(w).mean())
    d["gjr_fc"] = np.exp(d["lrv_m"]).fillna(2.0); d["u_gjr"] = np.log(d["rv_target"]) - np.log(d["gjr_fc"])
    iv_var = np.exp(lv + 0.25)                                                  # options know the latent variance, with a premium
    iv = pd.DataFrame({"date": idx, "iv30": np.sqrt(iv_var * 252) / 100, "skew_near": 0.03, "term_slope": 0.0})
    iv["sigma_day_implied"] = 100 * iv["iv30"] / math.sqrt(252)
    x = compare.join(iv, d)
    v = compare.vrp(x); assert v["ratio_implied_to_realised"] > 1 and v["share_days_vrp_positive"] > 0.5
    mz = compare.mincer_zarnowitz(x).set_index("forecast")
    assert mz.loc["implied (options)", "r2"] > mz.loc["GJR-GARCH", "r2"]
    lad = compare.ladder(x)
    t = lad[(lad.period == "test")].set_index("model")
    assert t.loc["+HAR +options", "qlike"] < t.loc["GJR-GARCH", "qlike"]
