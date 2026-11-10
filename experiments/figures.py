"""Figures for the README / write-up, from the saved results of experiments/event_reaction.py.

  python experiments/figures.py          → docs/figures/*.png

fig1  Reaction-day forecast error: GARCH → + filing metadata → + FinBERT, results announcements vs all filings
fig2  FinBERT's contribution by subset with 95% CIs (where it helps, where it doesn't)
fig3  Placebo: real FinBERT text vs 400 shuffles of the same company's other quarters
fig4  Every out-of-sample results day: actual volatility vs GARCH vs GARCH + metadata + FinBERT
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "experiments" / "event_reaction"
OUT = ROOT / "docs" / "figures"
# palette (validated reference palette): text inks, recessive grid, categorical slots in fixed order
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#77756f", "#e3e1dc"
GARCH, META, FINBERT = "#9b9a94", "#eb6834", "#2a78d6"            # baseline = neutral grey · slot 2 · slot 1
NAMES = {"B0": "GARCH only", "B1": "+ filing metadata", "B2": "+ FinBERT on the filing"}
CO = {"TCS": "TCS", "INFY": "Infosys", "WIPRO": "Wipro", "CIPLA": "Cipla", "DRREDDY": "Dr Reddy's", "SUNPHARMA": "Sun Pharma"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})


def note(fig, text):
    fig.text(0.0, -0.02, text, fontsize=8.8, color=MUTED, ha="left", va="top", wrap=True)


def fig1(pred: pd.DataFrame):
    rows = []
    for name, m in (("Results-related announcements", pred.group == "results"), ("All filings", pred.index == pred.index)):
        for k in ("B0", "B1", "B2"):
            rows.append((name, k, float(((pred.u_react - pred[f"u_hat_{k}"]) ** 2)[m].mean()), int(m.sum())))
    d = pd.DataFrame(rows, columns=["subset", "model", "mse", "n"])
    fig, ax = plt.subplots(figsize=(8.2, 3.9)); w = 0.26; groups = d.subset.unique()
    for i, k in enumerate(("B0", "B1", "B2")):
        v = d[d.model == k].set_index("subset").loc[groups, "mse"].values; x = np.arange(len(groups)) + (i - 1) * (w + 0.02)
        ax.bar(x, v, width=w, color=(GARCH, META, FINBERT)[i], label=NAMES[k], zorder=2)
        for xi, vi in zip(x, v): ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    for j, g in enumerate(groups):
        b1, b2 = (d[(d.subset == g) & (d.model == k)].mse.iloc[0] for k in ("B1", "B2"))
        ax.annotate(f"FinBERT: {100 * (b2 / b1 - 1):+.1f}%", xy=(j + (w + 0.02), b2), xytext=(j + 0.55, b1 + 0.28),
                    fontsize=10, color=FINBERT, fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-", color=FINBERT, lw=1))
    ax.set_xticks(range(len(groups)), [f"{g}\n(n = {d[d.subset == g].n.iloc[0]:,})" for g in groups], color=INK2)
    ax.set_ylabel("forecast error (MSE of the log-variance miss)"); ax.set_ylim(0, d.mse.max() * 1.32); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9.5)
    ax.set_title("Reading the announcement cuts the reaction-day forecast error on results news")
    note(fig, "Reaction-day realised variance vs the GJR-GARCH forecast made the evening before; out-of-sample 2023–2026, "
              "each year predicted by a model fitted on earlier years only. Lower is better.")
    fig.savefig(OUT / "fig1_results_day_error.png"); plt.close(fig)


def fig2(subs: pd.DataFrame):
    keep = ["results days", "· results release (Financial Results / Board outcome)", "· earnings-call transcript", "after-close filings", "all", "intraday filings", "excluding results days", "year 2023", "year 2024", "year 2025", "year 2026"]
    lab = {"results days": "Results-related announcements", "· results release (Financial Results / Board outcome)": "   · results releases",
           "· earnings-call transcript": "   · earnings-call transcripts", "after-close filings": "Filed after the close", "all": "All filings",
           "intraday filings": "Filed during trading", "excluding results days": "Everything except results"}
    s = subs.set_index("subset").loc[[k for k in keep if k in set(subs.subset)]].reset_index()
    s["lo"], s["hi"] = 100 * s.ci95_lo / s.B1_mse, 100 * s.ci95_hi / s.B1_mse
    fig, ax = plt.subplots(figsize=(8.2, 4.3)); y = np.arange(len(s))[::-1]
    for yi, r in zip(y, s.itertuples()):
        sig = r.hi < 0; c = FINBERT if sig else GARCH
        ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([r.change_pct], [yi], s=58, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(4, yi, f"{r.change_pct:+.1f}%", va="center", ha="left", fontsize=10, color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
        ax.text(10.5, yi, f"n = {r.n:,}", va="center", ha="left", fontsize=9, color=MUTED)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(y, [lab.get(k, k.replace("year", "Year")) for k in s.subset], color=INK2)
    ax.set_xlabel("change in forecast error from adding FinBERT (%, 95% CI)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(min(-30, s.lo.min() - 2), 16)
    ax.set_title("FinBERT helps where there is real news: results, released after the close")
    note(fig, "Blue: confidence interval excludes zero. Intervals: bootstrap over announcement dates (5,000 draws). "
              "Baseline includes filing type, company, company × results day, timing, recent stock and market volatility.")
    fig.savefig(OUT / "fig2_where_it_helps.png"); plt.close(fig)


def fig3(pl: pd.DataFrame, real: float):
    g = pl[pl.scheme.str.startswith("P2")].t1_mse
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.hist(g, bins=30, color=GARCH, edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(real, color=FINBERT, lw=2.4, zorder=3)
    ax.text(real, ax.get_ylim()[1] * 0.92, "  real FinBERT text\n  (better than all\n  400 shuffles)", color=FINBERT, fontsize=10, fontweight="bold", va="top")
    ax.text(g.mean(), ax.get_ylim()[1] * 1.0, f"shuffled text: same company, same filing type, another quarter", color=INK2, fontsize=9.5, ha="center", va="bottom")
    ax.set_xlabel("out-of-sample forecast error (MSE)"); ax.set_ylabel("shuffles"); ax.grid(axis="x", visible=False)
    ax.set_title("The signal is in what each announcement says, not who is announcing", pad=20)
    note(fig, "Placebo: FinBERT scores permuted within company × filing type, model refitted, 400 times.")
    fig.savefig(OUT / "fig3_placebo.png"); plt.close(fig)


def fig4(pred: pd.DataFrame):
    r = pred[pred.group == "results"].copy(); r["react"] = pd.to_datetime(r.react)
    for k in ("B0", "B2"): r[f"sig_{k}"] = np.sqrt(r[f"rv_hat_{k}"])
    r["sig_act"] = np.sqrt(r.rv_react)
    cos = [c for c in CO if c in set(r.symbol)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True); axes = axes.ravel()
    top = np.nanpercentile(r[["sig_act", "sig_B0", "sig_B2"]].values, 99) * 1.08
    for ax, c in zip(axes, cos):
        g = r[r.symbol == c].sort_values("react"); x = np.arange(len(g))
        for xi, a_, b_ in zip(x, g.sig_B0, g.sig_B2):
            ax.plot([xi, xi], [a_, b_], color=FINBERT, lw=1.2, alpha=0.55, zorder=1)
        ax.scatter(x, g.sig_B0, s=16, color=GARCH, zorder=2, label="GARCH forecast")
        ax.scatter(x, g.sig_B2, s=16, color=FINBERT, zorder=3, label="GARCH + metadata + FinBERT")
        ax.scatter(x, g.sig_act, s=30, marker="_", color=INK, linewidths=2, zorder=4, label="actual")
        ax.set_title(CO[c], fontsize=11); ax.set_ylim(0, top); ax.grid(axis="x", visible=False)
        yrs = g.react.dt.year.values; ticks = [i for i in range(len(g)) if i == 0 or yrs[i] != yrs[i - 1]]
        ax.set_xticks(ticks, [str(yrs[i]) for i in ticks])
    for ax in axes[len(cos):]: ax.axis("off")
    axes[0].set_ylabel("reaction-day volatility (daily %)"); axes[3].set_ylabel("reaction-day volatility (daily %)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    fig.suptitle("Every results-related announcement, 2023–2026: actual reaction vs GARCH vs GARCH + metadata + FinBERT",
                 x=0.0, ha="left", y=1.06, fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    note(fig, "Volatility = √(overnight gap² + Parkinson range); forecasts made the evening before the reaction day, "
              "out of sample. Lines show how far metadata + FinBERT moved each GARCH forecast — usually up, toward the realised "
              "reaction; it also over-shoots on some days (the error measure counts both).")
    fig.savefig(OUT / "fig4_results_days.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RES / "t1_predictions.csv"); subs = pd.read_csv(RES / "t1_subsets.csv"); pl = pd.read_csv(RES / "placebo.csv")
    real = float(((pred.u_react - pred.u_hat_B2) ** 2).mean())
    fig1(pred); fig2(subs); fig3(pl, real); fig4(pred)
    print("written:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
