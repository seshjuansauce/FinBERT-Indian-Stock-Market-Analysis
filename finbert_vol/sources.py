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
    """BSE corporate filings: axiom's research/data/filings.parquet and/or data/calls/bse_filings.parquet (from
    BSETranscriptFetcher). Keyed by symbol (TCS, CIPLA, …); an ISIN also works for the companies in config.ISIN."""

    CALL_WORDS = re.compile(r"earnings?\s+(?:conference\s+)?call|conference\s+call|analysts?\s+(?:/\s*investors?\s+)?call|"
                            r"investors?\s+call|post[- ]results?|results?\s+call|con-?call", re.I)
    COLS = ["symbol", "isin", "date", "announced_at", "after_close", "bse_category", "bse_subcategory", "bse_headline", "path"]

    def __init__(self, filings_path: Path | None, extra: list[Path] | None = None):
        from .config import ISIN
        to_sym = {v: k for k, v in ISIN.items()}
        frames = []
        if filings_path is not None and Path(filings_path).exists():
            d = pd.read_parquet(filings_path); d["symbol"] = d["isin"].map(to_sym).fillna(d["isin"]); frames.append(d)
        for e in extra or []:
            if Path(e).exists(): frames.append(pd.read_parquet(e))
        if not frames:
            raise FileNotFoundError(f"no filings: {filings_path} not found (set axiom_dir / AXIOM_DIR) and no fetched filings")
        df = pd.concat([f[[c for c in self.COLS if c in f.columns]] for f in frames], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        self.df = df.drop_duplicates(["symbol", "date", "bse_headline"]).reset_index(drop=True)

    @property
    def symbols(self) -> list[str]: return sorted(self.df["symbol"].dropna().unique())

    def company(self, key: str) -> pd.DataFrame:
        return self.df[(self.df["symbol"] == key) | (self.df["isin"] == key)].sort_values("date")

    def results(self, key: str) -> pd.DataFrame:
        c = self.company(key)
        return c[c["bse_subcategory"].isin(["Financial Results", "Outcome of Board Meeting"])]

    def transcripts(self, key: str) -> pd.DataFrame:
        """Earnings-call transcript filings: BSE's 'Earnings Call Transcript' subcategory, or (before it existed, 2021)
        a filing whose headline says it is a call transcript. Re-filings are resolved by CallScorer (first parsed call wins)."""
        from .bse_fetch import is_transcript
        c = self.company(key)
        t = c[[is_transcript(s or "", h or "") for s, h in zip(c["bse_subcategory"], c["bse_headline"])]]
        return t[t["path"].notna()].drop_duplicates("path").sort_values("date")

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

    def __init__(self, jsonl: Path, pdf_root: Path | list[Path] | None = None):
        roots = pdf_root if isinstance(pdf_root, (list, tuple)) else [pdf_root] if pdf_root else []
        self.jsonl, self.roots, self._cache = Path(jsonl), [Path(r) for r in roots], {}
        if self.jsonl.exists():
            for line in self.jsonl.open():
                j = json.loads(line); self._cache[j["p"]] = j["t"]

    def __contains__(self, p: str) -> bool: return p in self._cache

    def text(self, p: str) -> str:
        if p in self._cache: return self._cache[p]
        full = next((r / p for r in self.roots if (r / p).exists()), None)
        if full is None: return ""
        import pdfplumber
        with pdfplumber.open(full) as pdf:
            t = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        self._cache[p] = t
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl.open("a") as fh: fh.write(json.dumps({"p": p, "pages": None, "t": t}) + "\n")
        return t
