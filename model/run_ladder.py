"""The volatility ladder: does TCS news explain what GJR-GARCH misses?

  python model/run_ladder.py                      # rungs 0–1 + placebo (attention only)
  python model/run_ladder.py --sentiment A,C      # + rungs using news_daily_tcs_A.csv / news_daily_tcs_C.csv

Target   u_{t+1} = ln RV_{t+1} − ln h_{t+1|t}   (h = walk-forward GJR-GARCH forecast, model/build_features.py)
Model    OLS  u_{t+1} = α + βᵀx_t + ε, expanding window, refit every 21 trading days, features standardised on the
         training window only;  forecast  RV̂_{t+1} = h_{t+1|t} · exp(α̂ + β̂ᵀx_t + s²/2)   (lognormal mean correction)
Periods  2020–21 GARCH burn-in · 2022-07 → 2023-12 VALIDATION (all design choices) · 2024-01 → last news date TEST
Scoring  QLIKE = RV/RV̂ − ln(RV/RV̂) − 1 (lower is better), RMSE of ln RV, Mincer–Zarnowitz R²;
         Diebold–Mariano test on the daily QLIKE difference with Newey–West standard errors.
Rungs    R0 GJR raw · R0c GJR calibrated (intercept only) · R0b + HAR terms + NIFTY 500 RV
         R1 + TCS attention · P1 = R1 with attention rows shuffled within each year (placebo, 20 draws)
         R2_<tag> + sentiment features from FinBERT model <tag>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]; FEAT, OUT = ROOT / "data" / "features", ROOT / "results" / "ladder"
VAL, TEST = ("2022-07-01", "2023-12-31"), ("2024-01-01", None)
HAR = ["lrv_d", "lrv_w", "lrv_m", "lmkt_d", "lmkt_w"]
ATT = ["log_n", "log_primary", "results_day", "log_focus", "no_news"]
SENT = ["mag", "sent", "sent_min", "carry", "focus_x_carry", "focus_para_sent"]

def market_rv() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data" / "prices.csv"); p = p[p.Symbol == "^CRSLDX"].copy()
    p["date"] = pd.to_datetime(p.Date.str[:10]); p = p.set_index("date").sort_index()
    p = p[p.High > p.Low]
    rv = 1e4 * (np.log(p.Open / p.Close.shift(1)) ** 2 + np.log(p.High / p.Low) ** 2 / (4 * np.log(2)))
    return pd.DataFrame({"lmkt_d": np.log(rv), "lmkt_w": np.log(rv.rolling(5).mean())})

def load(sent_tags: list[str]) -> pd.DataFrame:
    px = pd.read_csv(FEAT / "price_features_tcs.csv", parse_dates=["date"]).set_index("date")
    nd = pd.read_csv(FEAT / "news_daily_tcs.csv", parse_dates=["date"]).set_index("date")
    nd["log_primary"] = np.log1p(nd.n_primary); nd["log_focus"] = np.log1p(nd.n_focus)
    d = px.join(market_rv(), how="left").join(nd[ATT], how="left")
    for tag in sent_tags:
        s = pd.read_csv(FEAT / f"news_daily_tcs_{tag}.csv", parse_dates=["date"]).set_index("date")
        d = d.join(s[SENT].add_suffix(f"_{tag}"), how="left")
    art = pd.read_csv(FEAT / "articles_tcs.csv", parse_dates=["date"])
    last_news = art.date.max()
    d = d.dropna(subset=["gjr_fc", "rv_target"] + HAR)
    return d[d.index <= last_news], last_news

def walk_forward(d: pd.DataFrame, feats: list[str], refit=21, min_train=120, start=VAL[0]) -> pd.Series:
    """Out-of-sample RV forecasts for rows ≥ start; each block's model sees rows strictly before the block."""
    y = d["u_gjr"].values; X = d[feats].values.astype(float) if feats else np.zeros((len(d), 0))
    out = pd.Series(np.nan, index=d.index); i0 = d.index.searchsorted(pd.Timestamp(start))
    for s in range(i0, len(d), refit):
        tr = slice(0, s)
        if s < min_train: continue
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Z = np.column_stack([np.ones(s), (X[tr] - mu) / sd])
        beta, *_ = np.linalg.lstsq(Z, y[tr], rcond=None)
        s2 = np.var(y[tr] - Z @ beta, ddof=Z.shape[1])
        e = min(s + refit, len(d))
        Zt = np.column_stack([np.ones(e - s), (X[s:e] - mu) / sd])
        out.iloc[s:e] = d["gjr_fc"].values[s:e] * np.exp(Zt @ beta + s2 / 2)
    return out

def qlike(rv, f): x = rv / f; return x - np.log(x) - 1

def dm_test(l1: np.ndarray, l2: np.ndarray) -> tuple[float, float]:
    """H0: equal mean loss. Returns (DM stat, two-sided p). Positive stat = model 1 has higher loss."""
    dlt = l1 - l2; T = len(dlt); lag = int(np.floor(4 * (T / 100) ** (2 / 9))); dc = dlt - dlt.mean()
    gamma = [np.sum(dc[k:] * dc[:T - k]) / T for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    stat = dlt.mean() / np.sqrt(var / T); return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))

def scores(d, f, mask):
    rv, ff = d.rv_target[mask].values, f[mask].values
    y, x = np.log(rv), np.log(ff); r2 = np.corrcoef(x, y)[0, 1] ** 2
    return {"n": int(mask.sum()), "qlike": float(qlike(rv, ff).mean()), "rmse_ln": float(np.sqrt(np.mean((y - x) ** 2))), "mz_r2": float(r2)}

def coef_table(d, feats, period=VAL):
    import statsmodels.api as sm
    m = (d.index >= period[0]) & (d.index <= period[1]); X = d.loc[m, feats]; X = (X - X.mean()) / X.std().replace(0, 1)
    res = sm.OLS(d.loc[m, "u_gjr"], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return pd.DataFrame({"beta (per 1 sd)": res.params, "t (HAC)": res.tvalues, "p": res.pvalues}).round(3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sentiment", default="", help="comma list of tags, e.g. A,C")
    ap.add_argument("--placebo-draws", type=int, default=20); a = ap.parse_args()
    tags = [t for t in a.sentiment.split(",") if t]
    d, last_news = load(tags); OUT.mkdir(parents=True, exist_ok=True)
    rungs = {"R0 GJR raw": None, "R0c GJR calibrated": [], "R0b +HAR +market": HAR, "R1 +attention": HAR + ATT}
    for t in tags: rungs[f"R2_{t} +sentiment({t})"] = HAR + ATT + [f"{c}_{t}" for c in SENT]
    F = {k: (d["gjr_fc"].copy() if v is None else walk_forward(d, v)) for k, v in rungs.items()}
    # placebo: shuffle the attention rows across days within each calendar year (keeps level/seasonality, breaks timing)
    rng = np.random.default_rng(0); plac = []
    for _ in range(a.placebo_draws):
        dp = d.copy()
        for _, idx in dp.groupby(dp.index.year).groups.items():
            dp.loc[idx, ATT] = dp.loc[idx, ATT].values[rng.permutation(len(idx))]
        plac.append(walk_forward(dp, HAR + ATT))
    F["P1 placebo (shuffled attention)"] = pd.concat(plac, axis=1).mean(axis=1)
    per = {"validation": (d.index >= VAL[0]) & (d.index <= VAL[1]), "test": d.index >= TEST[0]}
    per["test · results days"] = per["test"] & (d.results_day > 0)
    per["test · no-news days"] = per["test"] & (d.no_news > 0)
    names = list(F); rows = []; base = "R0b +HAR +market"
    for pname, m in per.items():
        m = m & np.all([F[k].notna() for k in names], axis=0)
        for i, k in enumerate(names):
            sc = scores(d, F[k], m); row = {"period": pname, "rung": k, **sc}
            for ref in ([names[i - 1]] if i > 0 and not k.startswith("P1") else []) + ([base] if k not in (base,) and i > 2 else []):
                st, p = dm_test(qlike(d.rv_target[m].values, F[k][m].values), qlike(d.rv_target[m].values, F[ref][m].values))
                row[f"DM vs {ref.split()[0]}"] = f"{st:+.2f} (p={p:.3f})"
            rows.append(row)
    res = pd.DataFrame(rows)
    pd.concat({k: v for k, v in F.items()}, axis=1).assign(rv_target=d.rv_target, results_day=d.results_day).to_csv(OUT / "forecasts.csv")
    coefs = coef_table(d, HAR + ATT)
    # figure: cumulative QLIKE gain of R1 over R0b in the test period
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m = per["test"]; gain = (qlike(d.rv_target[m], F[base][m]) - qlike(d.rv_target[m], F["R1 +attention"][m])).cumsum()
        gp = (qlike(d.rv_target[m], F[base][m]) - qlike(d.rv_target[m], F["P1 placebo (shuffled attention)"][m])).cumsum()
        fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(gain.index, gain.values, label="R1 attention vs R0b")
        ax.plot(gp.index, gp.values, label="placebo vs R0b", ls="--")
        for t in d.index[m & (d.results_day > 0)]: ax.axvline(t, color="grey", lw=0.4, alpha=0.5)
        ax.axhline(0, color="black", lw=0.6); ax.set_ylabel("cumulative QLIKE reduction"); ax.set_title("TCS: test period (grey = results-news days)")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT / "cumulative_gain_test.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)
    fmt = lambda df: df.to_markdown(index=False) if hasattr(df, "to_markdown") and _has_tabulate() else df.to_string(index=False)
    md = ["# Volatility ladder — TCS", f"\nNews corpus ends {last_news.date()}; test period capped there. Placebo = mean of {a.placebo_draws} within-year shuffles.\n"]
    for pname in per:
        md += [f"\n## {pname}\n", "```", res[res.period == pname].drop(columns="period").round(4).to_string(index=False), "```"]
    md += ["\n## R1 coefficients, validation period (standardised features, HAC t-stats)\n", "```", coefs.to_string(), "```"]
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    res.to_csv(OUT / "ladder.csv", index=False)
    print("\n".join(md))

def _has_tabulate():
    try: import tabulate  # noqa: F401
    except ImportError: return False
    return True

if __name__ == "__main__":
    main()
