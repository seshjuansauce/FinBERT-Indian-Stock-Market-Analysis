"""Score every BSE filing in axiom's research table with FinBERT-IN (target-aware, temperature-calibrated).

  python scripts/score_filings.py [--axiom ~/Desktop/Projects/axiom]     # ~5 min on Apple-silicon GPU, ~25 min on 2 CPU cores
Out  data/events/filing_finbert.parquet   id, isin, symbol, date, fb_z_*, fb_p_*, fb_s, fb_label
Text = BSE headline + ". " + summary, or the summary alone when the headline is a stub ("Enclosed", "Press release").
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from finbert_vol.config import ISIN  # noqa: E402
from finbert_vol.models import SentimentModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--axiom", default=os.environ.get("AXIOM_DIR", str(ROOT.parent / "axiom")))
    ap.add_argument("--model", default=str(ROOT / "models" / "finbert-in-sentiment")); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    f = pd.read_parquet(Path(a.axiom) / "research" / "data" / "filings.parquet")
    f["symbol"] = f["isin"].map({v: k for k, v in ISIN.items()})
    head, summ = f["bse_headline"].fillna("").str.strip(), f["summary"].fillna("").str.strip()
    f["model_text"] = np.where(head.str.len() > 25, head + ". " + summ, summ)
    m = SentimentModel.from_settings(a.model, ROOT / "results" / "sentiment" / "calibration.json", "C", a.device); m.max_len = 160
    t0, out = time.time(), []
    for i in range(0, len(f), 256):
        ch = f.iloc[i:i + 256]; out.append(m.predict(ch["model_text"].tolist(), ch["symbol"].tolist()))
        print(f"{i + len(ch)}/{len(f)}  {time.time() - t0:.0f}s", flush=True)
    res = pd.concat([f[["id", "isin", "symbol", "date", "announced_at"]].reset_index(drop=True), pd.concat(out, ignore_index=True).add_prefix("fb_")], axis=1)
    p = ROOT / "data" / "events" / "filing_finbert.parquet"; p.parent.mkdir(parents=True, exist_ok=True); res.to_parquet(p, index=False)
    print(p, res.shape, res["fb_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
