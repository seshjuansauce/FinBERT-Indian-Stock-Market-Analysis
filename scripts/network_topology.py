"""Headline co-mention network over the NIFTY 50: topology report + TCS monthly centrality series.

  python scripts/network_topology.py                      # all of data/hf/raw_dataset/*_raw.csv
  python scripts/network_topology.py --min-weight 3       # drop pairs seen in fewer than 3 headlines
  python scripts/network_topology.py --max-names 4 --min-weight 10   # recommended: no market roundups, real ties only

Why the filters: a "Sensex: Reliance, TCS, HDFC Bank, Infosys…" roundup links every company it names, so at
min-weight 1 the graph is ~complete (density ≈ 0.93) and unweighted measures carry no information.

Out  results/network/graph_metrics.json      graph-level topology (full period, and per year)
     results/network/node_metrics.csv        per-company centralities, community, sector
     results/network/edges.csv               weighted edge list (for Gephi / later GNN work)
     data/features/network_tcs_monthly.csv   TCS position month by month (future model feature —
                                             use it lagged one month to stay causal)
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finbert_vol.network import CoMentionNetwork  # noqa: E402


def headlines(pattern: str) -> pd.DataFrame:
    fr = [pd.read_csv(f, usecols=["date", "title", "url"]) for f in sorted(glob.glob(pattern))]
    d = pd.concat(fr, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d.dropna(subset=["date", "title"]).drop_duplicates("url").drop_duplicates(["date", "title"])
    return d.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "data/hf/raw_dataset/*_raw.csv"))
    ap.add_argument("--min-weight", type=int, default=1); ap.add_argument("--focus", default="TCS")
    ap.add_argument("--max-names", type=int, default=0, help="drop headlines naming more than this many companies (0 = keep all)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    out = ROOT / "results" / "network" / (a.tag or "all"); out.mkdir(parents=True, exist_ok=True)
    d = headlines(a.glob); net = CoMentionNetwork(); M = net.mentions(d["title"])
    multi = (M.sum(1) >= 2) & ((M.sum(1) <= a.max_names) if a.max_names else True)
    print(f"{len(d):,} headlines; {int(M.any(axis=1).sum()):,} name ≥1 NIFTY company; {int(multi.sum()):,} name ≥2 (these form edges)")

    E = net.edges(M[multi]); E.sort_values("weight", ascending=False).to_csv(out / "edges.csv", index=False)
    G = net.graph(E, list(M.columns), a.min_weight)
    gm = {"full": net.graph_metrics(G)}
    for y, idx in d.groupby(d.date.dt.year).groups.items():
        sub = M.loc[idx]; sub = sub[(sub.sum(1) <= a.max_names) if a.max_names else slice(None)]; gm[str(y)] = {"headlines": len(sub), **net.graph_metrics(net.graph(net.edges(sub[sub.sum(1) >= 2]), list(M.columns), a.min_weight))}
    (out / "graph_metrics.json").write_text(json.dumps(gm, indent=1, default=float))
    nm = net.node_metrics(G); nm.insert(0, "headlines", M.sum()); nm.to_csv(out / "node_metrics.csv")

    mon = net.monthly(d.loc[multi, "date"], M[multi], focus=a.focus, min_weight=a.min_weight)
    mon.to_csv(ROOT / "data" / "features" / f"network_{a.focus.lower()}_monthly{('_' + a.tag) if a.tag else ''}.csv", index=False)

    f = a.focus; rank = lambda c: int(nm[c].rank(ascending=False, method="min")[f])
    print("\nGraph (full period):", {k: round(v, 3) if isinstance(v, float) else v for k, v in gm["full"].items()})
    print(f"\n{f}: degree {nm.loc[f, 'degree']} (rank {rank('degree')}), strength {nm.loc[f, 'strength']} (rank {rank('strength')}), "
          f"betweenness rank {rank('betweenness')}, eigenvector rank {rank('eigenvector')}, community {nm.loc[f, 'community']} "
          f"= {sorted(nm.index[nm.community == nm.loc[f, 'community']])}")
    print("\nTop 10 by strength:\n", nm.head(10)[["headlines", "degree", "strength", "betweenness", "eigenvector", "community", "sector"]].round(3).to_string())
    print(f"\nStrongest {f} ties:\n", E[(E.source == f) | (E.target == f)].sort_values("weight", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
