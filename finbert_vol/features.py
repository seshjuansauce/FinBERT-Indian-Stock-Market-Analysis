"""FeatureBuilder: one row per trading day t (information known before the open of t+1), target RV_{t+1}."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

CALL_FEATURES = ["tone_answers", "tone_analysts", "tone_gap_qa_minus_prepared", "neg_share_qa", "hedge_rate_answers",
                 "forward_tone", "dispersion_qa", "magnitude_qa"]


class FeatureBuilder:
    def __init__(self, features_dir: Path, target: str):
        self.dir, self.target = Path(features_dir), target

    # ---- prices (rung 0 inputs, from model/build_features.py) ----
    def prices(self) -> pd.DataFrame:
        p = self.dir / f"price_features_{self.target.lower()}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run: python model/build_features.py --symbol {self.target}.NS")
        return pd.read_csv(p, parse_dates=["date"]).set_index("date")

    # ---- news ----
    def news_daily(self, articles: pd.DataFrame, td: pd.DatetimeIndex, sent_col: str | None) -> pd.DataFrame:
        from score_corpus import daily
        return daily(articles, td, sent_col)

    # ---- calls: a transcript's features are live for `horizon` trading days from its publication row ----
    @staticmethod
    def call_daily(calls: pd.DataFrame, td: pd.DatetimeIndex, horizon: int = 10) -> pd.DataFrame:
        td = pd.DatetimeIndex(td).sort_values()
        out = pd.DataFrame(0.0, index=td, columns=CALL_FEATURES + ["call_live", "days_since_call"])
        out["days_since_call"] = np.nan
        if calls is None or calls.empty: return out
        pos = td.searchsorted(pd.DatetimeIndex(calls["published"]).normalize(), side="right") - 1
        for (_, c), i in zip(calls.iterrows(), pos):
            if i < 0: continue
            for k in range(horizon):
                j = i + k
                if j >= len(td): break
                out.iloc[j, out.columns.get_loc("call_live")] = 1.0
                out.iloc[j, out.columns.get_loc("days_since_call")] = k
                for f in CALL_FEATURES:
                    v = c.get(f, np.nan); out.iloc[j, out.columns.get_loc(f)] = 0.0 if pd.isna(v) else float(v)
        return out

    def build(self, articles: pd.DataFrame | None, sent_col: str | None, calls: pd.DataFrame | None,
              flags: pd.DataFrame | None, horizon: int = 10) -> pd.DataFrame:
        px = self.prices(); td = px.index
        parts = [px]
        if articles is not None:
            nd = self.news_daily(articles, td, sent_col)
            nd["log_primary"] = np.log1p(nd["n_primary"]); nd["log_focus"] = np.log1p(nd["n_focus"])
            parts.append(nd.drop(columns=[c for c in nd.columns if c in px.columns]))
        if calls is not None: parts.append(self.call_daily(calls, td, horizon))
        if flags is not None: parts.append(flags.reindex(td).fillna(0.0))
        table = pd.concat(parts, axis=1)
        table.index.name = "date"
        return table
