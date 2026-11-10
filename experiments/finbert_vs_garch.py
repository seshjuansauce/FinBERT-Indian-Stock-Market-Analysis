"""Experiment: does FinBERT news sentiment improve GJR-GARCH volatility forecasts for TCS — and which part of it?

  python experiments/finbert_vs_garch.py                  # ~1–2 min
  python experiments/finbert_vs_garch.py --placebo 200 --boot 5000

Question, split into the pieces a reviewer would ask about
  Q1  Does the full GARCH + FinBERT model beat GARCH?                                 M4  vs M0
  Q2  …or is the gain just from adding past volatility (HAR) terms?                   M4  vs M1
  Q3  Does FinBERT *sentiment* add anything beyond *counting* articles?               M4  vs M2   ← the key test
  Q4  Does domain adaptation matter (FinBERT-IN vs off-the-shelf ProsusAI FinBERT)?   M4  vs M4a
  Q5  Is the sentiment signal real, or would any noise of the same shape do?          M4  vs placebo distribution
  Q6  Where does it help — results days, news days, no-news days, by year?            subsets

Models (all forecast RV_{t+1} = h_{t+1|t} · exp(x_t'β̂ + s²/2); β̂ refit every 21 days on data before each block)
  M0   GJR-GARCH(1,1,1)-t                         x = ∅ (raw h)
  M1   + HAR + market                             x = ln RV_t, ln RV̄_5, ln RV̄_22, ln RV_mkt, ln RV̄_mkt,5
  M2   + news volume (no FinBERT)                 x = M1 + log articles, log primary, results-news day, log roundups, no-news
  M3   + FinBERT-IN sentiment, no volume          x = M1 + SENT(FinBERT-IN, calibrated)
  M4   + volume + FinBERT-IN sentiment  (full)    x = M2 + SENT(FinBERT-IN, calibrated)
  M4a  + volume + ProsusAI FinBERT sentiment      x = M2 + SENT(ProsusAI, calibrated)
  SENT = mean, |mean|, min, 30-day decayed carry, roundup×carry, roundup-paragraph sentiment  (fixed before seeing test data)

Evaluation
  Periods   validation 2022-07 → 2023-12 (all design choices made here) · test 2024-01 → end of the news corpus
  Loss      QLIKE = RV/F − ln(RV/F) − 1   (robust to noise in the RV proxy; lower is better)
  Tests     Diebold–Mariano (Newey–West) on daily loss differences; 95% moving-block bootstrap CI (block 10 days)
            of the mean loss difference; Model Confidence Set (Hansen–Lunde–Nason, 10%) over M0–M4a
  Placebo   M4 refit with the sentiment columns shuffled across days *within each year* (keeps their distribution
            and the real news-volume columns; breaks only the link between what was said and when) — p = share of
            draws that do at least as well as the real sentiment
Out  results/experiments/finbert_vs_garch/{summary.md, models.csv, contrasts.csv, subsets.csv, placebo.csv, forecasts.csv}
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from run_ladder import ATT, HAR, SENT, TEST, VAL, dm_test, load, qlike, walk_forward  # noqa: E402

OUT = ROOT / "results" / "experiments" / "finbert_vs_garch"
S_IN, S_PR = [f"{c}_Ct" for c in SENT], [f"{c}_At" for c in SENT]       # calibrated FinBERT-IN (C) / ProsusAI (A)
MODELS = {
    "M0 GJR-GARCH": None,
    "M1 +HAR+market": HAR,
    "M2 +news volume": HAR + ATT,
    "M3 +FinBERT-IN sentiment (no volume)": HAR + S_IN,
    "M4 +volume +FinBERT-IN": HAR + ATT + S_IN,
    "M4a +volume +ProsusAI FinBERT": HAR + ATT + S_PR,
}
CONTRASTS = [  # (model, reference, question)
    ("M4 +volume +FinBERT-IN", "M0 GJR-GARCH", "Q1 full model vs GARCH"),
    ("M4 +volume +FinBERT-IN", "M1 +HAR+market", "Q2 beyond past volatility (HAR)"),
    ("M4 +volume +FinBERT-IN", "M2 +news volume", "Q3 FinBERT sentiment beyond article counts"),
    ("M3 +FinBERT-IN sentiment (no volume)", "M1 +HAR+market", "Q3b sentiment alone, over HAR"),
    ("M4 +volume +FinBERT-IN", "M4a +volume +ProsusAI FinBERT", "Q4 domain-adapted vs off-the-shelf FinBERT"),
    ("M2 +news volume", "M1 +HAR+market", "(ref) article counts over HAR"),
]


def block_bootstrap_ci(dl: np.ndarray, B: int, block: int = 10, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(dl) by moving-block bootstrap (keeps autocorrelation in daily loss differences)."""
    rng = np.random.default_rng(seed); n = len(dl); k = int(np.ceil(n / block)); means = np.empty(B)
    starts = np.arange(n - block + 1)
    for b in range(B):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = dl[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--placebo", type=int, default=100); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    d, last_news = load(["Ct", "At"])
    d[S_IN + S_PR] = d[S_IN + S_PR].fillna(0.0); d[ATT] = d[ATT].fillna(0.0)
    F = {k: (d["gjr_fc"].copy() if v is None else walk_forward(d, v)) for k, v in MODELS.items()}
    ok = np.all([F[k].notna().values for k in F], axis=0)
    per = {"validation": ok & (d.index >= VAL[0]) & (d.index <= VAL[1]), "test": ok & (d.index >= TEST[0])}
    L = {k: qlike(d.rv_target.values, F[k].values) for k in F}

    # ---- 1. every model, every period
    rows = []
    for p, m in per.items():
        for k in F:
            y, x = np.log(d.rv_target.values[m]), np.log(F[k].values[m])
            rows.append({"period": p, "model": k, "n": int(m.sum()), "qlike": L[k][m].mean(),
                         "vs_M0_pct": 100 * (L[k][m].mean() / L["M0 GJR-GARCH"][m].mean() - 1),
                         "rmse_lnRV": float(np.sqrt(np.mean((y - x) ** 2))), "mz_r2": float(np.corrcoef(x, y)[0, 1] ** 2)})
    models = pd.DataFrame(rows)

    # ---- 2. the questions: DM + block-bootstrap CI of the mean QLIKE difference (negative = model better)
    rows = []
    for p, m in per.items():
        for mod, ref, q in CONTRASTS:
            dl = L[mod][m] - L[ref][m]; st, pv = dm_test(L[mod][m], L[ref][m]); lo, hi = block_bootstrap_ci(dl, a.boot)
            rows.append({"period": p, "question": q, "model": mod, "reference": ref, "mean_dQLIKE": dl.mean(),
                         "pct": 100 * dl.mean() / L[ref][m].mean(), "ci95_lo": lo, "ci95_hi": hi, "dm": st, "p": pv,
                         "verdict": "better" if hi < 0 else "worse" if lo > 0 else "no significant difference"})
    contrasts = pd.DataFrame(rows)

    # ---- 3. Model Confidence Set on the test period
    mcs_txt = ""
    try:
        from arch.bootstrap import MCS
        m = per["test"]; losses = pd.DataFrame({k: L[k][m] for k in F})
        mcs = MCS(losses, size=0.10, reps=a.boot, block_size=10, method="R", seed=0); mcs.compute()
        mcs_txt = f"Model Confidence Set (test, 90%): {sorted(mcs.included)}  ·  eliminated: {sorted(mcs.excluded)}"
        pv = mcs.pvalues.iloc[:, 0] if isinstance(mcs.pvalues, pd.DataFrame) else mcs.pvalues
        pv.rename("mcs_p").rename_axis("model").reset_index().to_csv(OUT / "mcs.csv", index=False)
        mcs_txt += "  ·  p-values: " + ", ".join(f"{k.split()[0]} {v:.2f}" for k, v in pv.items())
    except Exception as e:  # noqa: BLE001
        mcs_txt = f"MCS skipped: {e}"

    # ---- 4. placebo: shuffle FinBERT sentiment within year, keep everything else
    rng = np.random.default_rng(1); pl = []
    for i in range(a.placebo):
        dp = d.copy()
        for _, idx in dp.groupby(dp.index.year).groups.items():
            dp.loc[idx, S_IN] = dp.loc[idx, S_IN].values[rng.permutation(len(idx))]
        fp = walk_forward(dp, HAR + ATT + S_IN); lp = qlike(d.rv_target.values, fp.values)
        pl.append({"draw": i, **{p: lp[m].mean() for p, m in per.items()}})
    placebo = pd.DataFrame(pl)
    real = {p: L["M4 +volume +FinBERT-IN"][m].mean() for p, m in per.items()}
    plac_p = {p: float((placebo[p] <= real[p]).mean()) for p in per}

    # ---- 5. where: subsets of the test period
    t = per["test"]; subsets = {"all test days": t, "results-news days": t & (d.results_day > 0), "news days (≥1 article)": t & (d.no_news == 0),
                                "no-news days": t & (d.no_news > 0), "strongly negative news (sent < −0.3)": t & (d["sent_Ct"] < -0.3),
                                "strongly positive news (sent > 0.3)": t & (d["sent_Ct"] > 0.3)}
    for y in sorted(set(d.index[t].year)): subsets[f"year {y}"] = t & (d.index.year == y)
    rows = []
    for name, m in subsets.items():
        if m.sum() < 5: continue
        r = {"subset": name, "n": int(m.sum())}
        for k in ("M0 GJR-GARCH", "M1 +HAR+market", "M2 +news volume", "M4 +volume +FinBERT-IN"): r[k.split()[0]] = L[k][m].mean()
        st, pv = dm_test(L["M4 +volume +FinBERT-IN"][m], L["M2 +news volume"][m]) if m.sum() > 20 else (np.nan, np.nan)
        r["M4_vs_M2_pct"] = 100 * (r["M4"] / r["M2"] - 1); r["dm_M4_vs_M2_p"] = pv
        rows.append(r)
    subs = pd.DataFrame(rows)

    # ---- outputs
    models.to_csv(OUT / "models.csv", index=False); contrasts.to_csv(OUT / "contrasts.csv", index=False)
    subs.to_csv(OUT / "subsets.csv", index=False); placebo.to_csv(OUT / "placebo.csv", index=False)
    pd.DataFrame(F).assign(rv_target=d.rv_target).to_csv(OUT / "forecasts.csv")
    f4 = lambda x: f"{x:.4f}"
    md = ["# Does FinBERT improve GARCH? — TCS", "",
          f"Data: walk-forward forecasts, validation {VAL[0]} → {VAL[1]}, test {TEST[0]} → {last_news.date()} (end of news corpus). "
          f"Loss: QLIKE (lower is better). Negative differences favour the first model.", "",
          "## Models", "", models.round(4).to_markdown(index=False) if _md() else models.round(4).to_string(index=False), "",
          "## The questions (mean daily QLIKE difference, 95% block-bootstrap CI, Diebold–Mariano)", "",
          (contrasts.round(4).to_markdown(index=False) if _md() else contrasts.round(4).to_string(index=False)), "",
          f"## Placebo ({a.placebo} within-year shuffles of the FinBERT-IN sentiment columns)", "",
          *[f"- {p}: real M4 QLIKE {f4(real[p])}; placebo mean {f4(placebo[p].mean())} (5–95%: {f4(placebo[p].quantile(.05))}–{f4(placebo[p].quantile(.95))}); "
            f"share of placebos at least as good = **{plac_p[p]:.2f}**" for p in per], "",
          f"## {mcs_txt}", "", "## Where (test period)", "",
          subs.round(4).to_markdown(index=False) if _md() else subs.round(4).to_string(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(md)); print("\n".join(md))


def _md():
    try: import tabulate  # noqa: F401
    except ImportError: return False
    return True


if __name__ == "__main__":
    main()
