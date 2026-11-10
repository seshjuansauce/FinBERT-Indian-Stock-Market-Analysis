"""Recompute every headline number in the README from the committed result files — no data download, no model.

  python experiments/verify_claims.py

Reads only:
  results/experiments/event_reaction/t1_predictions.csv   one row per announcement (out of sample): realised miss u, GARCH
                                                          forecast, and each model's prediction
  results/experiments/event_reaction/placebo.csv          placebo refits
  results/experiments/finbert_vs_garch/models.csv         daily experiment
Each claim is printed with the formula it comes from; the script fails if a recomputed number drifts from the README.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "experiments"
CLAIMS = {  # README value → tolerance
    "results_related_pct": (-13.5, 0.05), "all_filings_pct": (-2.0, 0.05),
}


def mse(u, uh): return float(np.mean((u - uh) ** 2))


def date_bootstrap(g: pd.DataFrame, B=5000, seed=0):
    """95% CI of the % change in MSE (B2 vs B1), resampling announcement dates with replacement."""
    rng = np.random.default_rng(seed); g = g.assign(e1=(g.u_react - g.u_hat_B1) ** 2, e2=(g.u_react - g.u_hat_B2) ** 2)
    by = g.groupby("react")[["e1", "e2"]].agg(["sum", "count"]); s1, s2, n = by[("e1", "sum")].values, by[("e2", "sum")].values, by[("e1", "count")].values
    out = []
    for _ in range(B):
        i = rng.integers(0, len(n), len(n)); out.append(100 * (s2[i].sum() / s1[i].sum() - 1))
    return np.percentile(out, [2.5, 97.5])


def main():
    p = pd.read_csv(R / "event_reaction" / "t1_predictions.csv")
    print("Announcement-level experiment  (u = ln RV_reaction − ln h_GARCH; error = mean squared error of the predicted u)\n")
    print(f"{'subset':44s} {'n':>5s} {'GARCH':>7s} {'+meta':>7s} {'+FinBERT':>9s} {'FinBERT effect':>15s}  95% CI")
    subsets = {"results-related announcements": p.group == "results",
               "  · results releases (Fin. Results / Board)": p.top_sub.isin(["Financial Results", "Outcome of Board Meeting"]) if "top_sub" in p else None,
               "  · earnings-call transcripts": (p.top_sub == "Earnings Call Transcript") if "top_sub" in p else None,
               "all filings": p.index == p.index, "all except results-related": p.group != "results"}
    got = {}
    for name, m in subsets.items():
        if m is None or m.sum() == 0: continue
        g = p[m]; b0, b1, b2 = (mse(g.u_react, g[f"u_hat_{k}"]) for k in ("B0", "B1", "B2")); pct = 100 * (b2 / b1 - 1); lo, hi = date_bootstrap(g)
        print(f"{name:44s} {len(g):5d} {b0:7.3f} {b1:7.3f} {b2:9.3f} {pct:+14.1f}%  [{lo:+.1f}%, {hi:+.1f}%]")
        got[name] = pct
    pl = pd.read_csv(R / "event_reaction" / "placebo.csv"); real = mse(p.u_react, p.u_hat_B2)
    print(f"\nPlacebos (real error {real:.4f}):")
    for s, g in pl.groupby("scheme"):
        print(f"  {s:18s} {len(g)} refits, mean error {g.t1_mse.mean():.4f}; refits at least as good as the real text: {(g.t1_mse <= real).sum()}  (p < {1 / (len(g) + 1):.4f})")
    d = pd.read_csv(R / "finbert_vs_garch" / "models.csv"); t = d[d.period == "test"].set_index("model")
    print("\nDaily experiment (TCS, test 2024 → Mar 2026, QLIKE):", {k.split()[0]: round(v, 4) for k, v in t.qlike.items()})

    ok = abs(got["results-related announcements"] - CLAIMS["results_related_pct"][0]) <= CLAIMS["results_related_pct"][1] and \
         abs(got["all filings"] - CLAIMS["all_filings_pct"][0]) <= CLAIMS["all_filings_pct"][1]
    print("\nREADME claims reproduced:", "YES" if ok else "NO — numbers differ from README")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
