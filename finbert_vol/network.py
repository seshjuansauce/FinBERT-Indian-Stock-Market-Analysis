"""Company co-mention network from news headlines: who is written about together, and how central TCS is.

Nodes  = the 50 NIFTY 50 companies (headline aliases below; ambiguous names are guarded, e.g. HDFC Bank ≠ HDFC Life,
         Tata Steel ≠ TCS, Reliance ≠ Reliance Power, M&M ≠ Tech Mahindra ≠ Kotak Mahindra).
Edges  = two companies named in the same headline; weight = number of such headlines.

Basic topology (graph level): nodes, edges, density, mean degree / strength, average clustering, transitivity,
degree assortativity, communities (greedy modularity) and modularity, largest component, diameter.
Node level: degree, strength, degree / betweenness / eigenvector centrality, PageRank, clustering, community.
Time series: the same node measures for TCS month by month — the hook for later network features.
"""
from __future__ import annotations
import itertools
import re

import numpy as np
import pandas as pd

ALIASES = {
    "TCS": (r"\btcs\b(?! (?:rates?|deduction|provisions?|credit|refunds?|norms?|rules?|exemption|to be (?:collected|levied|deducted)"
            r"|collect\w*|levied|on (?:sale|foreign|lrs|remittances?|overseas)))(?<!\btax collected at source \(tcs)|tata consultancy"),
    "INFY": r"\binfosys\b|\binfy\b", "WIPRO": r"\bwipro\b", "HCLTECH": r"\bhcl ?tech(?:nologies)?\b", "TECHM": r"\btech ?mahindra\b|\btechm\b",
    "RELIANCE": r"\breliance\b(?! (?:capital|power|infra\w*|home|communications|general|nippon|retail ventures|jio))|\bril\b",
    "HDFCBANK": r"\bhdfc bank\b", "ICICIBANK": r"\bicici bank\b", "SBIN": r"\bsbi\b(?! (?:life|card|mutual|funds?|general|caps))|state bank of india",
    "AXISBANK": r"\baxis bank\b", "KOTAKBANK": r"\bkotak(?: mahindra)? bank\b|\bkotak\b(?! (?:mutual|securities|institutional|life|alternate))",
    "BAJFINANCE": r"\bbajaj finance\b", "BAJAJFINSV": r"\bbajaj finserv\b", "BAJAJ-AUTO": r"\bbajaj auto\b",
    "BHARTIARTL": r"\bbharti airtel\b|\bairtel\b", "LT": r"\bl ?& ?t\b|larsen (?:&|and) toubro", "ITC": r"\bitc\b",
    "HINDUNILVR": r"\bhul\b|hindustan unilever", "MARUTI": r"\bmaruti\b", "M&M": r"\bm ?& ?m\b|mahindra (?:&|and) mahindra",
    "TMPV": r"\btata motors\b", "TATASTEEL": r"\btata steel\b", "TATACONSUM": r"\btata consumer\b", "TITAN": r"\btitan\b",
    "TRENT": r"\btrent\b", "SUNPHARMA": r"\bsun pharma\w*", "DRREDDY": r"\bdr\.? ?reddy'?s?\b", "CIPLA": r"\bcipla\b",
    "APOLLOHOSP": r"\bapollo hospitals?\b", "MAXHEALTH": r"\bmax health\w*", "NESTLEIND": r"\bnestle\b", "ASIANPAINT": r"\basian paints\b",
    "ULTRACEMCO": r"\bultratech\b", "GRASIM": r"\bgrasim\b", "HINDALCO": r"\bhindalco\b", "JSWSTEEL": r"\bjsw steel\b",
    "COALINDIA": r"\bcoal india\b", "ONGC": r"\bongc\b", "NTPC": r"\bntpc\b", "POWERGRID": r"\bpower ?grid\b",
    "ADANIENT": r"\badani enterprises\b|\badani ent\b", "ADANIPORTS": r"\badani ports\b", "EICHERMOT": r"\beicher\b",
    "ETERNAL": r"\bzomato\b|\beternal ltd\b|\beternal limited\b", "INDIGO": r"\bindigo\b|\binterglobe\b", "BEL": r"\bbharat electronics\b|\bbel\b",
    "JIOFIN": r"\bjio financial\b|\bjfsl\b", "SHRIRAMFIN": r"\bshriram finance\b", "SBILIFE": r"\bsbi life\b", "HDFCLIFE": r"\bhdfc life\b",
}
SECTOR = {"TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT", "HDFCBANK": "Banks", "ICICIBANK": "Banks", "SBIN": "Banks",
          "AXISBANK": "Banks", "KOTAKBANK": "Banks", "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "SHRIRAMFIN": "Financials",
          "JIOFIN": "Financials", "SBILIFE": "Financials", "HDFCLIFE": "Financials", "RELIANCE": "Energy", "ONGC": "Energy", "COALINDIA": "Energy",
          "NTPC": "Energy", "POWERGRID": "Energy", "ADANIENT": "Industrials", "ADANIPORTS": "Industrials", "LT": "Industrials", "BEL": "Industrials",
          "INDIGO": "Industrials", "MARUTI": "Autos", "M&M": "Autos", "TMPV": "Autos", "BAJAJ-AUTO": "Autos", "EICHERMOT": "Autos",
          "SUNPHARMA": "Health", "DRREDDY": "Health", "CIPLA": "Health", "APOLLOHOSP": "Health", "MAXHEALTH": "Health",
          "ITC": "Consumer", "HINDUNILVR": "Consumer", "NESTLEIND": "Consumer", "TATACONSUM": "Consumer", "TITAN": "Consumer",
          "TRENT": "Consumer", "ASIANPAINT": "Consumer", "ETERNAL": "Consumer", "BHARTIARTL": "Telecom", "TATASTEEL": "Materials",
          "JSWSTEEL": "Materials", "HINDALCO": "Materials", "ULTRACEMCO": "Materials", "GRASIM": "Materials"}


class CoMentionNetwork:
    def __init__(self, aliases: dict[str, str] | None = None):
        self.aliases = aliases or ALIASES
        self._rx = {k: re.compile(v, re.I) for k, v in self.aliases.items()}

    def mentions(self, titles: pd.Series) -> pd.DataFrame:
        """Boolean matrix: headline × company."""
        t = titles.fillna("")
        return pd.DataFrame({k: t.str.contains(rx) for k, rx in self._rx.items()}, index=titles.index)

    @staticmethod
    def edges(M: pd.DataFrame) -> pd.DataFrame:
        X = M.to_numpy(dtype=np.int32); C = X.T @ X; names = M.columns
        rows = [(names[i], names[j], int(C[i, j])) for i, j in itertools.combinations(range(len(names)), 2) if C[i, j] > 0]
        return pd.DataFrame(rows, columns=["source", "target", "weight"])

    @staticmethod
    def graph(edges: pd.DataFrame, nodes: list[str], min_weight: int = 1):
        import networkx as nx
        G = nx.Graph(); G.add_nodes_from(nodes)
        for r in edges[edges.weight >= min_weight].itertuples():
            G.add_edge(r.source, r.target, weight=r.weight, distance=1.0 / r.weight)
        return G

    @staticmethod
    def graph_metrics(G) -> dict:
        import networkx as nx
        from networkx.algorithms import community as nxc
        n, m = G.number_of_nodes(), G.number_of_edges()
        deg = dict(G.degree()); strength = dict(G.degree(weight="weight"))
        giant = G.subgraph(max(nx.connected_components(G), key=len)) if m else nx.Graph()
        comms = list(nxc.greedy_modularity_communities(G, weight="weight")) if m else []
        return {"nodes": n, "edges": m, "density": nx.density(G), "mean_degree": float(np.mean(list(deg.values()))),
                "mean_strength": float(np.mean(list(strength.values()))), "avg_clustering": nx.average_clustering(G),
                "avg_clustering_weighted": nx.average_clustering(G, weight="weight"), "transitivity": nx.transitivity(G),
                "degree_assortativity": nx.degree_assortativity_coefficient(G) if m > 1 else np.nan,
                "giant_component_nodes": giant.number_of_nodes(), "diameter_giant": nx.diameter(giant) if giant.number_of_nodes() > 1 else 0,  # empty graph → 0
                "n_communities": len(comms), "modularity": nxc.modularity(G, comms, weight="weight") if comms else np.nan,
                "isolates": nx.number_of_isolates(G)}

    @staticmethod
    def node_metrics(G) -> pd.DataFrame:
        import networkx as nx
        from networkx.algorithms import community as nxc
        comms = list(nxc.greedy_modularity_communities(G, weight="weight")) if G.number_of_edges() else []
        cid = {v: i for i, c in enumerate(comms) for v in c}
        try: eig = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception: eig = {v: np.nan for v in G}
        d = pd.DataFrame({"degree": dict(G.degree()), "strength": dict(G.degree(weight="weight")),
                          "degree_centrality": nx.degree_centrality(G), "betweenness": nx.betweenness_centrality(G, weight="distance"),
                          "eigenvector": eig, "pagerank": nx.pagerank(G, weight="weight") if G.number_of_edges() else {v: np.nan for v in G},
                          "clustering": nx.clustering(G, weight="weight"), "community": cid})
        d["sector"] = d.index.map(SECTOR)
        return d.sort_values("strength", ascending=False)

    def monthly(self, dates: pd.Series, M: pd.DataFrame, focus: str = "TCS", min_weight: int = 1) -> pd.DataFrame:
        """Focus company's network position month by month (headlines of that month only)."""
        rows = []
        for mth, idx in pd.Series(M.index, index=pd.to_datetime(dates).dt.to_period("M").values).groupby(level=0):
            Mm = M.loc[idx.values]; G = self.graph(self.edges(Mm), list(M.columns), min_weight)
            nm = self.node_metrics(G); gm = self.graph_metrics(G)
            r = {"month": str(mth), "headlines": len(Mm), "focus_headlines": int(Mm[focus].sum()), "graph_edges": gm["edges"],
                 "graph_density": gm["density"]}
            for c in ["degree", "strength", "degree_centrality", "betweenness", "eigenvector", "pagerank", "clustering"]:
                r[f"{focus.lower()}_{c}"] = nm.loc[focus, c]
            nb = [n for n in G.neighbors(focus)] if focus in G else []
            r[f"{focus.lower()}_it_peer_share"] = (sum(SECTOR.get(n) == "IT" for n in nb) / len(nb)) if nb else np.nan
            rows.append(r)
        return pd.DataFrame(rows)
