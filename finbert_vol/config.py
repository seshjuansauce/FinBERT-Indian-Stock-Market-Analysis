"""Settings: one dataclass, optionally overridden by finbert_vol.yaml at the repo root (or a path you pass)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISIN = {"TCS": "INE467B01029", "INFY": "INE009A01021", "WIPRO": "INE075A01022", "HCLTECH": "INE860A01027", "TECHM": "INE669C01036",
        "CIPLA": "INE059A01026", "DRREDDY": "INE089A01031", "SUNPHARMA": "INE044A01036"}
SYMBOL = {"TCS": "TCS.NS", "INFY": "INFY.NS", "WIPRO": "WIPRO.NS", "HCLTECH": "HCLTECH.NS", "TECHM": "TECHM.NS"}
SYMBOL.update({s: f"{s}.NS" for s in ("CIPLA", "DRREDDY", "SUNPHARMA")})


@dataclass
class Settings:
    target: str = "TCS"                                         # company the forecast is for
    call_targets: list[str] = field(default_factory=lambda: ["TCS", "INFY", "WIPRO"])  # pooled for the call test; ["ALL"] = every company with transcripts + prices
    news_model: str = "models/finbert-in-sentiment"             # FinBERT-IN (news, target-aware)
    call_model: str = "models/finbert-in-call"                  # FinBERT-IN-call (falls back to news_model if absent)
    calibration: str = "results/sentiment/calibration.json"     # temperatures (key = model tag)
    axiom_dir: str = os.environ.get("AXIOM_DIR", str(ROOT.parent / "axiom"))
    transcripts_jsonl: str = os.environ.get("TRANSCRIPTS_JSONL", str(ROOT / "data" / "calls" / "transcripts.jsonl"))
    features_dir: str = "data/features"
    out_dir: str = "results/pipeline"
    validation: tuple[str, str] = ("2022-07-01", "2023-12-31")
    test_start: str = "2024-01-01"
    call_horizon_days: int = 10                                 # how long a transcript's features stay "live" (trading days)
    device: str | None = None                                   # mps / cuda / cpu; None = auto

    # ---- helpers ----
    def path(self, rel: str) -> Path:
        p = Path(rel).expanduser()
        return p if p.is_absolute() else ROOT / p

    @property
    def isin(self) -> str: return ISIN.get(self.target, self.target)

    @property
    def symbol(self) -> str: return SYMBOL.get(self.target, f"{self.target}.NS")

    @property
    def filings(self) -> Path: return Path(self.axiom_dir).expanduser() / "research" / "data" / "filings.parquet"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        s = cls()
        cfg = Path(path) if path else ROOT / "finbert_vol.yaml"
        if cfg.exists():
            import yaml
            data = yaml.safe_load(cfg.read_text()) or {}
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k not in names: raise ValueError(f"unknown setting {k!r} in {cfg}")
                setattr(s, k, tuple(v) if k == "validation" else v)
        return s
