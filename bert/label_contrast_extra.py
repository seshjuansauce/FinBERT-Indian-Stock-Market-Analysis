"""Extra training headlines containing contrast connectives (despite / even as / but …), 2020–2023 only,
not already in sample_train.csv / sample_test.csv.  -> data/labels/sample_contrast.csv"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "news")); sys.path.insert(0, str(ROOT / "bert"))
from entity import CONCESSIVE, ADVERSATIVE
from label_sample import pool
p = pd.concat([pool(t) for t in ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"]]); p = p[p["date"] < "2024-01-01"]
seen = set(pd.read_csv(ROOT / "data/labels/sample_train.csv")["id"]) | set(pd.read_csv(ROOT / "data/labels/sample_test.csv")["id"])
c = p[~p["id"].isin(seen) & (p["title"].str.contains(CONCESSIVE) | p["title"].str.contains(ADVERSATIVE))].copy()
c = c.drop_duplicates("id", keep="first")  # same headline, several targets: keep one (TCS first)
c["kind"] = c["title"].map(lambda t: "concessive" if CONCESSIVE.search(t) else "adversative")
c.sort_values("date").to_csv(ROOT / "data/labels/sample_contrast.csv", index=False)
print(len(c), c["target"].value_counts().to_dict(), c["kind"].value_counts().to_dict(), c["role"].value_counts().to_dict())
