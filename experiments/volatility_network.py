"""Volatility spillover network: Indian IT stocks, the NIFTY IT index and the NIFTY 50.

  python experiments/volatility_network.py        # needs data/prices.csv (+ price features for the results-day test)

Part A · Connectedness (Diebold–Yilmaz 2012, finbert_vol/spillover.py) on daily log realised variance
         static (full sample) → who transmits / receives volatility, net pairwise network
         rolling 250-day window → how connected the system is over time, and NIFTY IT's net role
Part B · Results-day spillover: on the reaction day of company j's results release, how far does company i's realised
         variance exceed its own GJR-GARCH forecast?   mean u_i on j's reaction days (i ≠ j) vs all other days
         → does one IT company's results move the others' volatility beyond what their own GARCH expects?
Out  results/network/spillover/*  ·  docs/figures/fig5_spillover_network.png, fig6_rolling_spillover.png, fig7_results_spillover.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from finbert_vol.spillover import Connectedness, realised_log_variance  # noqa: E402

OUT = ROOT / "results" / "network" / "spillover"; FIG = ROOT / "docs" / "figures"
NODES = {"TCS": "TCS.NS", "Infosys": "INFY.NS", "Wipro": "WIPRO.NS", "HCLTech": "HCLTECH.NS", "TechM": "TECHM.NS",
         "NIFTY IT": "^CNXIT", "NIFTY 50": "^NSEI"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#77756f", "#e3e1dc"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9b9a94"


def part_a(y: pd.DataFrame, md: list):
    c = Connectedness(horizon=10, max_lags=5)
    th = c.gfevd(y); tb, total = c.table(th); pw = c.pairwise_net(th)
    (100 * th).round(2).to_csv(OUT / "gfevd_matrix.csv"); tb.round(2).to_csv(OUT / "connectedness.csv"); pw.round(3).to_csv(OUT / "pairwise_net.csv", index=False)
    roll_tot, roll_net = c.rolling(y, window=250, step=5)
    pd.concat([roll_tot, roll_net.add_prefix("net_")], axis=1).round(3).to_csv(OUT / "rolling.csv")
    md += ["## A · Connectedness (full sample)", "",
           f"{y.index.min():%Y-%m-%d} → {y.index.max():%Y-%m-%d}, {len(y)} days, VAR({c.p_}) on ln RV, 10-day horizon, generalised FEVD.", "",
           f"**Total spillover index: {total:.1f} %** — share of all forecast-error variance that comes from other nodes.", "",
           "Share of each row's 10-day volatility forecast-error variance coming from each column (%):", "",
           (100 * th).round(1).to_markdown(), "",
           "| node | FROM others | TO others | NET (TO − FROM) | role |", "|---|---|---|---|---|"]
    for n, r in tb.sort_values("NET", ascending=False).iterrows():
        md.append(f"| {n} | {r.FROM:.1f} | {r.TO:.1f} | {r.NET:+.1f} | {'transmitter' if r.NET > 0 else 'receiver'} |")
    md += ["", "Strongest net pairwise flows (source → target, % of target's forecast-error variance):", "",
           pw.head(8).round(2).to_markdown(index=False), "",
           f"Rolling 250-day window: total spillover ranges {roll_tot.min():.1f}–{roll_tot.max():.1f} % "
           f"(lowest {roll_tot.idxmin():%Y-%m}, highest {roll_tot.idxmax():%Y-%m}).", ""]
    return th, tb, total, pw, roll_tot, roll_net


def part_b(md: list):
    """Results-day spillover among the IT companies with BSE results dates (TCS, Infosys, Wipro)."""
    ev = ROOT / "results" / "experiments" / "event_reaction" / "events.csv"
    if not ev.exists(): md.append("Part B skipped: events.csv not found"); return None
    e = pd.read_csv(ev, parse_dates=["date", "react"])
    rel = e[e.top_sub.isin(["Financial Results", "Outcome of Board Meeting"]) if "top_sub" in e else e.group.eq("results")]
    cos = {"TCS": "tcs", "Infosys": "infy", "Wipro": "wipro", "HCLTech": "hcltech", "TechM": "techm"}
    sym = {"TCS": "TCS", "Infosys": "INFY", "Wipro": "WIPRO"}
    react = {k: set(rel.loc[rel.symbol == v, "react"]) for k, v in sym.items()}
    px = {}
    for k, f in cos.items():
        p = ROOT / "data" / "features" / f"price_features_{f}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["date"]).set_index("date"); s = d["u_gjr"].copy(); s.index = d.index.map(lambda t: t)  # row t → RV_{t+1}
            nxt = pd.Series(list(d.index[1:]) + [pd.NaT], index=d.index); px[k] = pd.Series(s.values, index=nxt.values).dropna()  # keyed by the day RV realised
    rows = []
    any_res = set().union(*react.values())                   # every known results reaction day (TCS, Infosys, Wipro)
    for j, days in react.items():                            # reporter
        for i, u in px.items():                              # affected company
            u = u[u.index >= "2022-01-01"]
            own = react.get(i, set()) if i != j else set()   # drop the affected firm's OWN results days (same-day reporting)
            on = u.index.isin(list(days - own)); base = ~u.index.isin(list(any_res))   # baseline: no known IT results that day
            x, z = u[on], u[base]
            if on.sum() < 5: continue
            diff = x.mean() - z.mean(); se = np.sqrt(x.var(ddof=1) / len(x) + z.var(ddof=1) / len(z))
            rows.append({"reporter": j, "affected": i, "own": i == j, "n_days": int(on.sum()), "mean_u_on": x.mean(), "mean_u_other": z.mean(),
                         "excess_u": diff, "t": diff / se, "rv_ratio": float(np.exp(diff))})
    rb = pd.DataFrame(rows); rb.to_csv(OUT / "results_day_spillover.csv", index=False)
    piv = rb.pivot(index="affected", columns="reporter", values="excess_u").round(2)
    md += ["## B · Results-day spillover (does a peer's results day surprise my GARCH?)", "",
           "excess u = mean (ln RV − ln h_GARCH) on the reporter's results reaction days (excluding the affected firm's own results days) "
           "minus days with no known IT results, 2022 →. HCLTech and TechM own results dates are not in the filings table, so their "
           "column cannot exclude own-results overlap. "
           "exp(excess u) ≈ how many times larger realised variance was than the stock's own GARCH expected, relative to normal days.", "",
           "Rows = affected stock, columns = company reporting results:", "", piv.to_markdown(), "",
           rb.round(3).to_markdown(index=False), ""]
    return rb


def figures(th, tb, total, pw, roll_tot, roll_net, rb, y):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlesize": 12.5, "savefig.dpi": 200, "savefig.bbox": "tight"})
    # ---- fig5: network
    import networkx as nx
    G = nx.DiGraph(); G.add_nodes_from(th.index)
    thr = pw.net_pct.quantile(0.4)
    for r in pw.itertuples():
        if r.net_pct >= thr: G.add_edge(r.source, r.target, w=r.net_pct)
    pos = nx.circular_layout(sorted(G.nodes, key=lambda n: ["NIFTY 50", "NIFTY IT", "TCS", "Infosys", "Wipro", "HCLTech", "TechM"].index(n)))
    fig, ax = plt.subplots(figsize=(7.6, 6.6)); ax.axis("off")
    wmax = max(d["w"] for *_, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.6 + 5 * d["w"] / wmax, alpha=0.85,
                    shrinkA=26, shrinkB=26, mutation_scale=12 + 10 * d["w"] / wmax, connectionstyle="arc3,rad=0.08"))
    for n in G.nodes:
        net = tb.loc[n, "NET"]; c = BLUE if net > 0 else ORANGE
        ax.scatter(*pos[n], s=900 + 40 * abs(net), color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.text(pos[n][0], pos[n][1] - 0.2, f"{n}\nnet {net:+.1f}", ha="center", va="top", fontsize=10, color=INK, zorder=4)
    ax.set_title(f"Who transmits volatility to whom (total spillover {total:.0f} %)", fontsize=12.5, fontweight="bold", loc="left", color=INK)
    ax.scatter([], [], s=120, color=BLUE, label="net transmitter"); ax.scatter([], [], s=120, color=ORANGE, label="net receiver")
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    fig.text(0.02, 0.0, "Arrows: net pairwise spillover (Diebold–Yilmaz, generalised FEVD, 10-day horizon) of daily log realised variance; "
                        "width ∝ strength; weakest 40 % of pairs hidden. Node size ∝ |net|.", fontsize=8.5, color=MUTED, wrap=True)
    fig.savefig(FIG / "fig5_spillover_network.png"); plt.close(fig)
    # ---- fig6: rolling total + NIFTY IT / TCS net (two panels, one axis each)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    a1.plot(roll_tot.index, roll_tot.values, color=INK, lw=1.8); a1.set_ylabel("total spillover (%)"); a1.grid(color=GRID)
    a1.set_title("How connected IT volatility is, and who leads (rolling 250 trading days)")
    for n, c in (("NIFTY IT", BLUE), ("TCS", ORANGE), ("NIFTY 50", GREY)):
        a2.plot(roll_net.index, roll_net[n], color=c, lw=1.8, label=n)
        a2.text(roll_net.index[-1], roll_net[n].iloc[-1], f"  {n}", color=c, va="center", fontsize=9.5)
    a2.axhline(0, color=INK2, lw=0.8); a2.set_ylabel("net spillover (TO − FROM, %)"); a2.grid(color=GRID)
    a2.text(roll_net.index[0], a2.get_ylim()[1] * 0.92, "above 0 = transmitting", color=MUTED, fontsize=8.5, va="top")
    fig.text(0.0, -0.01, "Each point = connectedness estimated on the previous 250 trading days.", fontsize=8.5, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "fig6_rolling_spillover.png"); plt.close(fig)
    # ---- fig7: results-day spillover heatmap
    if rb is not None and len(rb):
        piv = rb.pivot(index="affected", columns="reporter", values="excess_u")
        tv = rb.pivot(index="affected", columns="reporter", values="t")
        rows = [r for r in ["TCS", "Infosys", "Wipro", "HCLTech", "TechM"] if r in piv.index]; cols = [c for c in ["TCS", "Infosys", "Wipro"] if c in piv.columns]
        piv, tv = piv.loc[rows, cols], tv.loc[rows, cols]
        fig, ax = plt.subplots(figsize=(6.4, 4.6)); vmax = np.nanmax(np.abs(piv.values))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
        ax.imshow(piv.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for a in range(len(rows)):
            for b in range(len(cols)):
                v, t = piv.values[a, b], tv.values[a, b]
                ax.text(b, a, f"{v:+.2f}\n×{np.exp(v):.1f}" + ("" if abs(t) >= 2 else "\n(n.s.)"), ha="center", va="center",
                        fontsize=9.5, color=INK, fontweight="bold" if abs(t) >= 2 else "normal")
        ax.set_xticks(range(len(cols)), [f"{c} reports" for c in cols]); ax.set_yticks(range(len(rows)), rows)
        ax.grid(False); ax.set_xlabel("results reaction day of …"); ax.set_ylabel("volatility surprise of …")
        ax.set_title("Results-day spillover: GARCH surprise when a peer reports")
        fig.text(0.0, -0.04, "Cell: mean (ln RV − ln h_GARCH) on the reporter's results reaction days minus other days, 2022 →; ×k = realised "
                             "variance k times its usual ratio to GARCH. Own-results days of the affected firm excluded (TCS/Infosys/Wipro). Bold: |t| ≥ 2.", fontsize=8.3, color=MUTED, wrap=True)
        fig.savefig(FIG / "fig7_results_spillover.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(ROOT / "data" / "prices.csv")
    y = realised_log_variance(prices, NODES); y = y[y.index >= "2020-03-01"]
    md = ["# Volatility spillover network — Indian IT", ""]
    th, tb, total, pw, roll_tot, roll_net = part_a(y, md)
    rb = part_b(md)
    figures(th, tb, total, pw, roll_tot, roll_net, rb, y)
    (OUT / "summary.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
