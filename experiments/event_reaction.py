"""Experiment 2: can FinBERT read a BSE announcement and predict the *initial* volatility reaction?

  python experiments/event_reaction.py            # needs data/events/filing_finbert.parquet + axiom filings + price features

Unit of analysis: one firm-day with at least one BSE filing (6 companies: TCS, Infosys, Wipro, Cipla, Dr Reddy's, Sun Pharma).
The day's filing with the highest-priority type represents it (results > management change > M&A > payout > analyst
meet > press release > general > other > procedural), as in axiom/research/features.py.

Reaction day R: the announcement's own trading day if it came before 15:30 IST, else the next trading day.

Targets
  T1  u_R = ln RV_R − ln h_R      how far realised variance on the reaction day overshoots the GJR-GARCH forecast made the
                                  evening before (h_R knows everything up to the previous close, nothing about the filing)
  T2  big_R = 1[|CAR(0,1)| / σ̂ > 2]  an abnormally large 2-day move, σ̂ = the firm's typical |CAR| on quiet filing days
                                  over the previous year (axiom sig_car_2); only reaction windows that contain a session

Feature sets (all known at announcement time)
  B0  nothing (T1: GARCH alone; T2: base rate)
  B1  metadata: filing type, company, company × results-day, after-close flag, ln σ̂ (firm's recent vol), stock RV (1d, 5d),
      market regime σ (EGARCH Nifty), # filings
  B2  B1 + FinBERT-IN on the filing text: s = P(pos) − P(neg), P(neg), P(pos), polarity 1 − P(neutral)
  B3  company + FinBERT only (can text stand in for the filing type?)

Protocol: expanding walk-forward by calendar year — fit on all events before 1 Jan Y, predict year Y (Y = 2023 … 2026).
  T1: OLS → out-of-sample MSE / R²_oos, and QLIKE of the implied RV forecast h·exp(û + s²/2) vs GARCH alone.
  T2: L2-logistic → Brier, log loss, AUC; Brier skill vs base rate.
  Uncertainty: 95% bootstrap CI of the loss difference B2 − B1, resampling dates (clusters).
  Placebos (each 200×; p = share of shuffles doing at least as well as the real text):
    P1 FinBERT columns shuffled within filing type × year  — keeps how upbeat each kind of filing is that year
    P2 FinBERT columns shuffled within filing type × company — same company, same event type, a different quarter's text
       (the strictest: rules out "the model just recognises which company is reporting")
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments" / "event_reaction"
SYM = {"INE467B01029": "TCS", "INE009A01021": "INFY", "INE075A01022": "WIPRO", "INE059A01026": "CIPLA",
       "INE089A01031": "DRREDDY", "INE044A01036": "SUNPHARMA"}
PRI = ["results", "mgmt", "mna", "payout", "analyst", "press", "general", "other", "procedural"]
FB = ["fb_s", "fb_p_negative", "fb_p_positive", "fb_polar"]
YEARS = [2023, 2024, 2025, 2026]


def build(axiom: Path, feat: Path) -> pd.DataFrame:
    f = pd.read_parquet(axiom / "research" / "data" / "filings.parquet")
    s = pd.read_parquet(ROOT / "data" / "events" / "filing_finbert.parquet")[["id"] + [c for c in ["fb_s", "fb_p_negative", "fb_p_positive", "fb_p_neutral"]]]
    f = f.merge(s, on="id", how="left"); f["fb_polar"] = 1 - f["fb_p_neutral"]
    f["symbol"] = f["isin"].map(SYM); f["pri"] = f["group"].map({g: i for i, g in enumerate(PRI)}).fillna(len(PRI))
    f["ts"] = pd.to_datetime(f["ts"]); f["date"] = pd.to_datetime(f["date"])
    rows = []
    for (sym, d), g in f.groupby(["symbol", "date"]):
        g = g.sort_values(["pri", "ts"]); top = g.iloc[0]
        rows.append({"symbol": sym, "date": d, "group": top["group"], "after_close": bool(top["after_close"]), "dow": int(top["dow"]),
                     "n_filings": len(g), "car_2": top["car_2"], "sig2": top["sig_car_2"], "sig_mkt": top["sig_mkt"], "ts": g["ts"].min(),
                     **{c: top[c] for c in FB},
                     "fb_absmax": g["fb_s"].abs().max()})                       # the most opinionated filing that day
    e = pd.DataFrame(rows)
    # reaction day + GARCH miss from the price features of that company
    out = []
    for sym, g in e.groupby("symbol"):
        px = pd.read_csv(feat / f"price_features_{sym.lower()}.csv", parse_dates=["date"]).set_index("date").sort_index()
        td = px.index
        same = td.searchsorted(g["date"].values, side="left")                  # first trading day ≥ date
        on_td = (same < len(td)) & (td[np.clip(same, 0, len(td) - 1)] == g["date"].values)
        r_idx = np.where(on_td & ~g["after_close"].values, same, np.where(on_td, same + 1, same))
        ok = r_idx < len(td)
        g = g[ok].copy(); r_idx = r_idx[ok]
        g["react"] = td[r_idx]
        prev = r_idx - 1; valid = prev >= 0
        g = g[valid]; prev = prev[valid]
        g["u_react"] = px["u_gjr"].values[prev]; g["h_react"] = px["gjr_fc"].values[prev]; g["rv_react"] = px["rv_target"].values[prev]
        g["lrv_d"] = px["lrv_d"].values[prev]; g["lrv_w"] = px["lrv_w"].values[prev]
        out.append(g)
    e = pd.concat(out, ignore_index=True)
    e["sar2"] = (e["car_2"] / e["sig2"]).abs()
    e["car2_valid"] = ~(e["after_close"] & e["dow"].isin([4, 5, 6])) & ~e["dow"].isin([5, 6])
    e["big"] = (e["sar2"] > 2).astype(float)
    e["year"] = e["react"].dt.year
    return e.sort_values("ts").reset_index(drop=True)


def design(e: pd.DataFrame, cols: str) -> pd.DataFrame:
    X = pd.DataFrame(index=e.index)
    if "meta" in cols:
        X = X.join(pd.get_dummies(e["group"], prefix="g", dtype=float)).join(pd.get_dummies(e["symbol"], prefix="c", dtype=float))
        X["after_close"] = e["after_close"].astype(float); X["ln_sig2"] = np.log(e["sig2"]); X["sig_mkt"] = e["sig_mkt"]
        X["ln_nfil"] = np.log(e["n_filings"]); X["lrv_d"] = e["lrv_d"]; X["lrv_w"] = e["lrv_w"]
        for s in sorted(e["symbol"].unique()):        # company-specific results-day effect (some firms' results move more)
            X[f"res_x_{s}"] = ((e["group"] == "results") & (e["symbol"] == s)).astype(float)
    if "company" in cols and "meta" not in cols: X = X.join(pd.get_dummies(e["symbol"], prefix="c", dtype=float))
    if "fb" in cols:
        for c in FB + ["fb_absmax"]: X[c] = e[c]
        X["fb_neg_x_results"] = e["fb_p_negative"] * (e["group"] == "results")
    return X.fillna(0.0)


SETS = {"B0": "", "B1 metadata": "meta", "B2 metadata+FinBERT": "meta fb", "B3 company+FinBERT": "company fb"}


def walk(e, target, kind, sets=SETS, fb_override=None):
    """Out-of-sample predictions for YEARS. kind: 'ols' (T1) or 'logit' (T2)."""
    from sklearn.linear_model import LogisticRegression
    pred = {k: pd.Series(np.nan, index=e.index) for k in sets}; s2 = pd.Series(np.nan, index=e.index)
    ee = e if fb_override is None else fb_override
    for k, cols in sets.items():
        X = design(ee, cols)
        for y in YEARS:
            tr, te = (e["year"] < y) & e[target].notna(), (e["year"] == y)
            if te.sum() == 0 or tr.sum() < 50: continue
            if not cols:
                m = e.loc[tr, target].mean(); pred[k][te] = 0.0 if kind == "ols" else m; continue
            Xtr, Xte = X[tr].values, X[te].values; mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            if kind == "ols":
                Z1 = np.column_stack([np.ones(len(Ztr)), Ztr]); b, *_ = np.linalg.lstsq(Z1, e.loc[tr, target].values, rcond=None)
                pred[k][te] = np.column_stack([np.ones(len(Zte)), Zte]) @ b
                if k == "B2 metadata+FinBERT": s2[te] = np.var(e.loc[tr, target].values - Z1 @ b)
            else:
                lr = LogisticRegression(C=0.5, max_iter=2000).fit(Ztr, e.loc[tr, target].values); pred[k][te] = lr.predict_proba(Zte)[:, 1]
    return pred


def cluster_ci(loss_diff: pd.Series, dates: pd.Series, B=2000, seed=0):
    rng = np.random.default_rng(seed); df = pd.DataFrame({"d": loss_diff.values, "c": dates.values}).groupby("c")["d"].agg(["sum", "count"])
    n = len(df); means = []
    for _ in range(B):
        i = rng.integers(0, n, n); means.append(df["sum"].values[i].sum() / df["count"].values[i].sum())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--placebo", type=int, default=200); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--axiom", default=os.environ.get("AXIOM_DIR", str(ROOT.parent / "axiom")))
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_auc_score
    e = build(Path(a.axiom), ROOT / "data" / "features"); e.to_csv(OUT / "events.csv", index=False)
    md = ["# Can FinBERT predict the initial reaction to a BSE announcement?", "",
          f"{len(e):,} firm-days with filings, {e.symbol.nunique()} companies, {e.date.min():%Y-%m} → {e.date.max():%Y-%m}. "
          f"Out-of-sample years {YEARS[0]}–{YEARS[-1]} (expanding walk-forward).", "",
          "Events by type: " + ", ".join(f"{k} {v}" for k, v in e.group.value_counts().items()), ""]

    # ---------- T1: GARCH miss on the reaction day ----------
    t1 = e[e["u_react"].notna()].reset_index(drop=True)
    P = walk(t1, "u_react", "ols"); oos = t1["year"].isin(YEARS) & P["B2 metadata+FinBERT"].notna()
    rows = []; y = t1.loc[oos, "u_react"]
    for k in P:
        err = (y - P[k][oos]) ** 2
        rv, h = t1.loc[oos, "rv_react"], t1.loc[oos, "h_react"]
        F = h * np.exp(P[k][oos] + (np.var(y - P[k][oos]) / 2 if k != "B0" else 0))
        q = rv / F - np.log(rv / F) - 1
        rows.append({"model": k, "n": int(oos.sum()), "mse_u": err.mean(), "r2_oos_vs_B0": 1 - err.mean() / ((y - 0) ** 2).mean(), "qlike_RV": q.mean()})
    t1tab = pd.DataFrame(rows)
    d12 = (y - P["B2 metadata+FinBERT"][oos]) ** 2 - (y - P["B1 metadata"][oos]) ** 2
    lo, hi = cluster_ci(d12, t1.loc[oos, "react"], a.boot)
    sub = t1[oos & (t1.group == "results")]
    res_rows = []
    for k in P:
        err = (sub["u_react"] - P[k][sub.index]) ** 2; res_rows.append({"model": k, "n": len(sub), "mse_u (results days)": err.mean()})
    sub_rows = []
    d_all = (t1["u_react"] - P["B2 metadata+FinBERT"]) ** 2 - (t1["u_react"] - P["B1 metadata"]) ** 2
    base_all = (t1["u_react"] - P["B1 metadata"]) ** 2
    masks = {"all": oos, "results days": oos & (t1.group == "results"), "excluding results days": oos & (t1.group != "results"),
             "after-close filings": oos & t1.after_close, "intraday filings": oos & ~t1.after_close}
    masks.update({f"year {yy}": oos & (t1.year == yy) for yy in YEARS})
    for name, m in masks.items():
        if m.sum() < 20: continue
        l, h = cluster_ci(d_all[m], t1.loc[m, "react"], a.boot)
        sub_rows.append({"subset": name, "n": int(m.sum()), "B1_mse": base_all[m].mean(), "B2_mse": base_all[m].mean() + d_all[m].mean(),
                         "change_pct": 100 * d_all[m].mean() / base_all[m].mean(), "ci95_lo": l, "ci95_hi": h})
    subs = pd.DataFrame(sub_rows); subs.to_csv(OUT / "t1_subsets.csv", index=False)
    md += ["## T1 · Reaction-day volatility beyond GARCH  (target u = ln RV_R − ln h_R; lower MSE / QLIKE is better)", "",
           t1tab.round(4).to_markdown(index=False), "",
           f"B2 − B1 (does FinBERT add to metadata?): ΔMSE = {d12.mean():+.4f}, 95% date-cluster CI [{lo:+.4f}, {hi:+.4f}]", "",
           "Results days only:", "", pd.DataFrame(res_rows).round(4).to_markdown(index=False), "",
           "FinBERT's contribution (B2 vs B1) by subset, with 95% date-cluster CIs:", "", subs.round(4).to_markdown(index=False), ""]

    # ---------- T2: abnormally large 2-day move ----------
    t2 = e[e["car2_valid"] & e["sar2"].notna() & e["sig2"].notna()].reset_index(drop=True)
    Q = walk(t2, "big", "logit"); oos2 = t2["year"].isin(YEARS) & Q["B2 metadata+FinBERT"].notna(); yb = t2.loc[oos2, "big"]
    rows = []
    for k in Q:
        p = Q[k][oos2].clip(1e-4, 1 - 1e-4); brier = ((yb - p) ** 2).mean()
        rows.append({"model": k, "n": int(oos2.sum()), "base_rate": yb.mean(), "brier": brier, "brier_skill_vs_B0": 1 - brier / ((yb - Q["B0"][oos2]) ** 2).mean(),
                     "log_loss": float(-(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean()), "auc": roc_auc_score(yb, p) if k != "B0" else 0.5})
    t2tab = pd.DataFrame(rows)
    b12 = (yb - Q["B2 metadata+FinBERT"][oos2]) ** 2 - (yb - Q["B1 metadata"][oos2]) ** 2
    lo2, hi2 = cluster_ci(b12, t2.loc[oos2, "date"], a.boot)
    md += ["## T2 · Probability of an abnormally large move  (|CAR(0,1)| > 2σ̂; lower Brier / log loss, higher AUC is better)", "",
           t2tab.round(4).to_markdown(index=False), "",
           f"B2 − B1 Brier: {b12.mean():+.5f}, 95% date-cluster CI [{lo2:+.5f}, {hi2:+.5f}]", ""]

    # ---------- placebos ----------
    rng = np.random.default_rng(7); pl = []
    real1 = ((y - P["B2 metadata+FinBERT"][oos]) ** 2).mean(); real2 = ((yb - Q["B2 metadata+FinBERT"][oos2]) ** 2).mean()
    one = {"B2 metadata+FinBERT": SETS["B2 metadata+FinBERT"]}
    for scheme, keys in (("P1 type×year", ["group", "year"]), ("P2 type×company", ["group", "symbol"])):
        for i in range(a.placebo):
            def shuf(df):
                d = df.copy()
                for _, idx in d.groupby(keys).groups.items():
                    d.loc[idx, FB + ["fb_absmax"]] = d.loc[idx, FB + ["fb_absmax"]].values[rng.permutation(len(idx))]
                return d
            p1 = walk(t1, "u_react", "ols", one, shuf(t1))["B2 metadata+FinBERT"]
            p2 = walk(t2, "big", "logit", one, shuf(t2))["B2 metadata+FinBERT"]
            pl.append({"scheme": scheme, "t1_mse": ((y - p1[oos]) ** 2).mean(), "t2_brier": ((yb - p2[oos2]) ** 2).mean()})
    pl = pd.DataFrame(pl); pl.to_csv(OUT / "placebo.csv", index=False)
    md += [f"## Placebos ({a.placebo} shuffles each of the FinBERT features)", ""]
    for scheme, g in pl.groupby("scheme"):
        md += [f"- {scheme} · T1: real MSE {real1:.4f}, placebo mean {g.t1_mse.mean():.4f}, share at least as good **{(g.t1_mse <= real1).mean():.3f}**"
               f" · T2: real Brier {real2:.5f}, placebo mean {g.t2_brier.mean():.5f}, share **{(g.t2_brier <= real2).mean():.3f}**"]
    md += [""]

    # ---------- direction (secondary): does tone predict the sign? ----------
    v = t2[t2.year.isin(YEARS)]
    from scipy.stats import spearmanr
    rho, pv = spearmanr(v["fb_s"], v["car_2"]); rr, pr = spearmanr(v.loc[v.group == "results", "fb_s"], v.loc[v.group == "results", "car_2"])
    rho_m, pm = spearmanr(v["fb_polar"], v["sar2"])
    md += ["## Direction vs magnitude (out-of-sample years, rank correlations)", "",
           f"- tone s vs CAR(0,1) (direction): ρ = {rho:+.3f} (p = {pv:.3f}); results days only ρ = {rr:+.3f} (p = {pr:.3f}, n = {int((v.group == 'results').sum())})",
           f"- polarity 1 − P(neutral) vs |CAR|/σ̂ (magnitude): ρ = {rho_m:+.3f} (p = {pm:.3f})", ""]
    t1tab.to_csv(OUT / "t1_reaction_vol.csv", index=False); t2tab.to_csv(OUT / "t2_large_move.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
