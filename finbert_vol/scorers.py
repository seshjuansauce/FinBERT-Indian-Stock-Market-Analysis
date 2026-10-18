"""Scorers: turn raw items (articles, transcripts, filings) into scored rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calls import CallFeatures, TranscriptParser
from .models import LABELS, SentimentModel
from .sources import BSEAnnouncements, Transcripts


class NewsScorer:
    """Headline sentiment (and roundup focus-text sentiment) for one target with a target-aware model."""

    def __init__(self, model: SentimentModel, target: str):
        self.model, self.target = model, target

    def score(self, articles: pd.DataFrame, tag: str | None = None) -> pd.DataFrame:
        tag = tag or self.model.name; art = articles.copy()
        head = self.model.predict(art["title"].tolist(), self.target)
        for l in LABELS: art[f"z_{tag}_{l}"] = head[f"z_{l}"].values
        art[f"sent_{tag}"] = head["s"].values
        has = art.get("focus_text", pd.Series("", index=art.index)).fillna("").str.len() > 40
        art[f"sent_{tag}_focus"] = np.nan
        if has.any():
            foc = self.model.predict(art.loc[has, "focus_text"].tolist(), self.target)
            art.loc[has, f"sent_{tag}_focus"] = foc["s"].values
            for l in LABELS: art.loc[has, f"zf_{tag}_{l}"] = foc[f"z_{l}"].values
        return art


class CallScorer:
    """Earnings-call transcripts → sentence sentiment → call-level features (one row per call)."""

    def __init__(self, model: SentimentModel, transcripts: Transcripts, parser: TranscriptParser | None = None):
        self.model, self.transcripts, self.parser = model, transcripts, parser or TranscriptParser()

    def score_one(self, text: str, target: str) -> tuple[pd.DataFrame, dict]:
        pc = self.parser.parse(text)
        sents = pc.sentences.copy()
        if len(sents):
            pr = self.model.predict(sents["sentence"].tolist(), target, sents["context"].tolist())
            sents = pd.concat([sents.reset_index(drop=True), pr], axis=1)
        feats = CallFeatures.compute(sents) if len(sents) else {}
        feats.update({"call_time": pc.call_time, "n_analysts": len(pc.analysts)})
        return sents, feats

    MIN_SENTENCES = 50      # fewer = garbled PDF or not a call (press release, press-conference-only transcript)
    SAME_CALL_DAYS = 20     # a second transcript this soon after a kept call is a re-filing / correction of it

    def score(self, filings: pd.DataFrame, target: str, after: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """filings: transcript rows (BSEAnnouncements.transcripts), any order. Returns (sentences, calls).
        Filings are taken in date order; one that does not parse as a call (< MIN_SENTENCES) is skipped, and one
        within SAME_CALL_DAYS of the last kept call is treated as the same call (first publication wins).
        after: date of the last call already scored for this company (incremental runs)."""
        all_s, calls, last = [], [], after
        for r in filings.sort_values("date").itertuples():
            d = pd.Timestamp(r.date)
            if last is not None and (d - last).days <= self.SAME_CALL_DAYS: continue
            text = self.transcripts.text(r.path)
            if not text or len(self.parser.parse(text).sentences) < self.MIN_SENTENCES: continue
            sents, feats = self.score_one(text, target)
            cid = f"{target}_{d:%Y%m%d}"; last = d
            all_s.append(sents.assign(call_id=cid, company=target))
            calls.append({"call_id": cid, "company": target, "published": d, "path": r.path, **feats})
        sents = pd.concat(all_s, ignore_index=True) if all_s else pd.DataFrame()
        return sents, pd.DataFrame(calls)


class AnnouncementScorer:
    """BSE filings → daily event flags on trading days (news timing rule: a filing dated d → row t = last trading day ≤ d)."""

    def __init__(self, bse: BSEAnnouncements):
        self.bse = bse

    @staticmethod
    def _rows(dates: pd.Series, td: pd.DatetimeIndex) -> pd.Series:
        pos = td.searchsorted(pd.DatetimeIndex(pd.to_datetime(dates)).normalize(), side="right") - 1
        return pd.Series(td[np.clip(pos, 0, None)], index=dates.index)[pos >= 0]

    def daily_flags(self, isin: str, td: pd.DatetimeIndex, lookahead_days: int = 15) -> pd.DataFrame:
        td = pd.DatetimeIndex(td).sort_values(); out = pd.DataFrame(index=td)
        res = self._rows(self.bse.results(isin)["date"], td)
        out["results_filing"] = out.index.isin(res.values).astype(float)
        intim = self._rows(self.bse.call_intimations(isin)["date"], td)
        # a call intimation stays "upcoming" until the next results filing (at most `lookahead_days` calendar days)
        up = pd.Series(0.0, index=td)
        res_sorted = np.sort(res.values)
        for d in intim.values:
            nxt = res_sorted[res_sorted >= d]
            end = nxt[0] if len(nxt) and (pd.Timestamp(nxt[0]) - pd.Timestamp(d)).days <= lookahead_days else pd.Timestamp(d) + pd.Timedelta(days=lookahead_days)
            up[(up.index >= d) & (up.index <= end)] = 1.0
        out["call_upcoming"] = up.values
        pub = self._rows(self.bse.transcripts(isin)["date"], td)
        out["transcript_published"] = out.index.isin(pub.values).astype(float)
        out.index.name = "date"
        return out
