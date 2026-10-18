"""Pipeline: wires sources → scorers → features → volatility model → outputs."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .features import CALL_FEATURES, FeatureBuilder
from .models import SentimentModel
from .scorers import AnnouncementScorer, CallScorer, NewsScorer
from .sources import BSEAnnouncements, NewsCorpus, Transcripts
from .volatility import ATT, HAR, Evaluator, VolatilityModel


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or Settings.load()
        self.out = self.s.path(self.s.out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.feat_dir = self.s.path(self.s.features_dir)

    # ---------- models ----------
    def news_model(self) -> SentimentModel:
        return SentimentModel.from_settings(self.s.path(self.s.news_model), self.s.path(self.s.calibration), "C", self.s.device)

    def call_model(self) -> SentimentModel:
        p = self.s.path(self.s.call_model)
        if not p.exists():
            print(f"[call model] {p} not found — using the news model (FinBERT-IN) with call inputs as a stand-in")
            m = self.news_model(); m.name = "C_on_calls"; return m
        return SentimentModel.from_settings(p, self.s.path(self.s.calibration), "CALL", self.s.device)

    # ---------- steps ----------
    def score_news(self, rebuild: bool = False) -> pd.DataFrame:
        corpus = NewsCorpus(self.feat_dir, self.s.target); art = corpus.load(rebuild)
        art = NewsScorer(self.news_model(), self.s.target).score(art, tag="C")
        corpus.save(art); return art

    def transcripts(self) -> Transcripts:
        return Transcripts(Path(self.s.transcripts_jsonl), [Path(self.s.axiom_dir).expanduser(), self.s.path('.')])

    def bse(self) -> BSEAnnouncements:
        """axiom filings + transcripts fetched from BSE (data/calls/bse_filings.parquet), if either exists."""
        fp = self.s.filings if self.s.filings.exists() else None
        return BSEAnnouncements(fp, [self.s.path('data/calls/bse_filings.parquet')])

    def call_universe(self, targets: list[str] | None = None) -> list[str]:
        t = targets or self.s.call_targets
        if [x.upper() for x in t] != ['ALL']: return list(t)
        bse = self.bse()
        return [s for s in bse.symbols if len(bse.transcripts(s)) and (self.feat_dir / f'price_features_{s.lower()}.csv').exists()]

    def score_calls(self, targets: list[str] | None = None, incremental: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score every transcript of the targets. incremental: keep calls already scored by the same model, score the rest."""
        bse = self.bse(); model = self.call_model(); scorer = CallScorer(model, self.transcripts())
        cdir = self.s.path("data/calls"); cdir.mkdir(parents=True, exist_ok=True)
        ps, pc = cdir / "call_sentences_scored.csv", cdir / "calls.csv"
        old_s, old_c = (pd.read_csv(ps), pd.read_csv(pc, parse_dates=["published"])) if incremental and ps.exists() and pc.exists() else (None, None)
        if old_c is not None and ("model" not in old_c or (old_c["model"] != model.name).any()): old_s = old_c = None
        done = set(old_c["path"]) if old_c is not None else set()
        S, Cs = ([old_s], [old_c]) if old_c is not None else ([], [])
        for t in self.call_universe(targets):
            f = bse.transcripts(t); f = f[~f["path"].isin(done)]
            prev = old_c.loc[old_c["company"] == t, "published"] if old_c is not None else pd.Series(dtype="datetime64[ns]")
            f = f[f["date"] > prev.max()] if len(prev) else f       # incremental: only filings after the last scored call
            if f.empty: continue
            sents, calls = scorer.score(f, t, prev.max() if len(prev) else None)
            if len(calls): S.append(sents); Cs.append(calls.assign(model=model.name))
            print(f"[calls] {t}: {len(calls)} new calls, {len(sents)} sentences", flush=True)
            sents_all, calls_all = pd.concat(S, ignore_index=True), pd.concat(Cs, ignore_index=True)
            sents_all.to_csv(ps, index=False); calls_all.to_csv(pc, index=False)        # checkpoint per company
        return (pd.concat(S, ignore_index=True), pd.concat(Cs, ignore_index=True)) if Cs else (pd.DataFrame(), pd.DataFrame())

    def load_calls(self) -> pd.DataFrame | None:
        p = self.s.path("data/calls/calls.csv")
        return pd.read_csv(p, parse_dates=["published"]) if p.exists() else None

    def flags(self, target: str | None = None) -> pd.DataFrame | None:
        t = target or self.s.target
        try:
            bse = self.bse()
        except FileNotFoundError as e:
            print(f"[flags] skipped: {e}"); return None
        td = FeatureBuilder(self.feat_dir, t).prices().index
        return AnnouncementScorer(bse).daily_flags(t, td)

    def build_features(self, target: str | None = None, sent_col: str = "sent_Ct") -> pd.DataFrame:
        t = target or self.s.target; fb = FeatureBuilder(self.feat_dir, t)
        art_path = self.feat_dir / f"articles_{t.lower()}.csv"
        art = pd.read_csv(art_path, parse_dates=["date"]) if art_path.exists() else None
        if art is not None and sent_col not in art: sent_col = next((c for c in ("sent_C", "sent_A") if c in art), None)
        calls = self.load_calls(); calls = calls[calls.company == t] if calls is not None else None
        table = fb.build(art, sent_col, calls, self.flags(t), self.s.call_horizon_days)
        table.to_csv(self.feat_dir / f"model_table_{t.lower()}.csv"); return table

    def rungs(self, table: pd.DataFrame) -> dict[str, list[str] | None]:
        r = {"R0 GJR": None, "R0b +HAR +market": HAR, "R1 +attention": HAR + ATT}
        if "call_upcoming" in table: r["R1e +announcement flags"] = HAR + ATT + ["call_upcoming", "results_filing"]
        if "call_live" in table and table["call_live"].sum() > 0:
            r["R3 +call features"] = HAR + ATT + ["call_live"] + [f for f in CALL_FEATURES if f in table]
        return r

    def evaluate(self, table: pd.DataFrame | None = None) -> pd.DataFrame:
        table = table if table is not None else pd.read_csv(self.feat_dir / f"model_table_{self.s.target.lower()}.csv", parse_dates=["date"]).set_index("date")
        fc = VolatilityModel(self.rungs(table), start=self.s.validation[0]).forecasts(table)
        masks = {"test · results days": table.get("results_day", pd.Series(0, index=table.index)) > 0}
        if "call_live" in table: masks["test · call windows"] = table["call_live"] > 0
        res = Evaluator(self.s.validation, self.s.test_start).table(fc, masks, baseline="R1 +attention")
        fc.to_csv(self.out / "forecasts.csv"); res.to_csv(self.out / "evaluation.csv", index=False)
        return res

    def call_window_panel(self, horizons=(1, 5, 10)) -> pd.DataFrame:
        """One row per call: mean GARCH miss u over the h trading days after publication, for each company."""
        calls = self.load_calls()
        if calls is None: raise FileNotFoundError("run score_calls first")
        rows = []
        for t, g in calls.groupby("company"):
            px = FeatureBuilder(self.feat_dir, t).prices(); td = px.index
            pos = td.searchsorted(pd.DatetimeIndex(g["published"]).normalize(), side="right") - 1
            for (_, c), i in zip(g.iterrows(), pos):
                row = c.to_dict()
                for h in horizons:
                    w = px["u_gjr"].iloc[i:i + h] if i >= 0 else pd.Series(dtype=float)
                    row[f"u_{h}d"] = float(w.mean()) if w.notna().sum() == h else np.nan
                rows.append(row)
        return pd.DataFrame(rows)

    def predict_latest(self, feats: list[str] | None = None) -> dict:
        table = pd.read_csv(self.feat_dir / f"model_table_{self.s.target.lower()}.csv", parse_dates=["date"]).set_index("date")
        res = VolatilityModel({}).latest(table, feats or HAR + ATT); res["target"] = self.s.target
        (self.out / "prediction_latest.json").write_text(json.dumps(res, indent=1)); return res
