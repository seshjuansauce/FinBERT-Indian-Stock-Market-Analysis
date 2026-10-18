"""Data sources: the news corpus, BSE announcements (from the axiom filings table) and earnings-call transcripts."""
from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd


class NewsCorpus:
    """TCS-relevant articles from the HF news corpus (cached at data/features/articles_<t>.csv by news/score_corpus.py)."""

    def __init__(self, features_dir: Path, target: str = "TCS"):
        self.path = Path(features_dir) / f"articles_{target.lower()}.csv"; self.target = target

    def load(self, rebuild: bool = False) -> pd.DataFrame:
        if self.path.exists() and not rebuild:
            return pd.read_csv(self.path, parse_dates=["date"])
        from score_corpus import articles          # duckdb scan of data/hf/raw_dataset
        df = articles(self.target); self.path.parent.mkdir(parents=True, exist_ok=True); df.to_csv(self.path, index=False)
        return df

    def save(self, df: pd.DataFrame): df.to_csv(self.path, index=False)


class BSEAnnouncements:
    """BSE corporate filings (axiom research/data/filings.parquet): results, call intimations, transcripts."""

    CALL_WORDS = re.compile(r"earnings?\s+(?:conference\s+)?call|conference\s+call|analysts?\s+(?:/\s*investors?\s+)?call|"
                            r"investors?\s+call|post[- ]results?|results?\s+call|con-?call", re.I)
    COLS = ["isin", "date", "announced_at", "after_close", "bse_category", "bse_subcategory", "bse_headline", "path"]

    def __init__(self, filings_path: Path):
        self.path = Path(filings_path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found — set axiom_dir in finbert_vol.yaml or AXIOM_DIR")
        df = pd.read_parquet(self.path)
        self.df = df[[c for c in self.COLS if c in df.columns]].copy()
        self.df["date"] = pd.to_datetime(self.df["date"])

    def company(self, isin: str) -> pd.DataFrame:
        return self.df[self.df["isin"] == isin].sort_values("date")

    def results(self, isin: str) -> pd.DataFrame:
        c = self.company(isin)
        return c[c["bse_subcategory"].isin(["Financial Results", "Outcome of Board Meeting"])]

    def transcripts(self, isin: str) -> pd.DataFrame:
        c = self.company(isin)
        return c[c["bse_subcategory"] == "Earnings Call Transcript"]

    def call_intimations(self, isin: str) -> pd.DataFrame:
        """Filings announcing an earnings call ahead of time: an Analyst/Investor Meet or Board Meeting filing whose
        headline mentions the call, or any Analyst/Investor Meet filed in the 15 days before a results filing."""
        c = self.company(isin)
        meet = c[c["bse_subcategory"].isin(["Analyst / Investor Meet", "Board Meeting"])]
        by_words = meet[meet["bse_headline"].fillna("").str.contains(self.CALL_WORDS)]
        res = self.results(isin)["date"].sort_values().values
        def before_results(d):
            nxt = res[res >= d.to_datetime64()]
            return len(nxt) > 0 and (pd.Timestamp(nxt[0]) - d).days <= 15
        by_time = meet[(meet["bse_subcategory"] == "Analyst / Investor Meet") & meet["date"].map(before_results)]
        return pd.concat([by_words, by_time]).drop_duplicates().sort_values("date")


class Transcripts:
    """Transcript text by filing path: from a JSONL cache ({p, pages, t}), else extracted from the PDF with pdfplumber."""

    def __init__(self, jsonl: Path, pdf_root: Path | None = None):
        self.jsonl, self.pdf_root, self._cache = Path(jsonl), Path(pdf_root) if pdf_root else None, {}
        if self.jsonl.exists():
            for line in self.jsonl.open():
                j = json.loads(line); self._cache[j["p"]] = j["t"]

    def __contains__(self, p: str) -> bool: return p in self._cache

    def text(self, p: str) -> str:
        if p in self._cache: return self._cache[p]
        if self.pdf_root is None or not (self.pdf_root / p).exists(): return ""
        import pdfplumber
        with pdfplumber.open(self.pdf_root / p) as pdf:
            t = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        self._cache[p] = t
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl.open("a") as fh: fh.write(json.dumps({"p": p, "pages": None, "t": t}) + "\n")
        return t
