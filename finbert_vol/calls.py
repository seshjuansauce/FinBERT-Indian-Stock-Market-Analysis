"""Earnings-call structure: parse a transcript into speaker turns and sentences, and turn sentence sentiment into
call-level features. What is specific to calls (vs news headlines):

  • who speaks — management (CEO/CFO/IR) vs analysts vs moderator
  • where — scripted prepared remarks vs unscripted Q&A (the Q&A is where information leaks: Matsumoto et al. 2011)
  • hedging / uncertainty language (Loughran–McDonald uncertainty words; Price et al. 2012 on call tone)
  • forward-looking statements (outlook, pipeline, next quarter) — the only part about the future
  • analyst tone — sceptical questions are themselves a signal (Chen, Nagar & Schoenfeld 2018)
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPANY = {"TCS": "tcs", "INFY": "infosys", "WIPRO": "wipro", "HCLTECH": "hcltech", "TECHM": "tech mahindra"}

START = re.compile(r"^\s*(?:Moderator|Operator)\s*:|ladies and gentlemen", re.I | re.M)
SPEAKER = re.compile(r"^(?P<name>[A-Z][A-Za-z\.\'’\- ]{1,48}?)\s*:\s+(?P<rest>.*)$")
NOT_SPEAKER = re.compile(r"^(?:Note|Sub|Encl|Tel|Fax|Symbol|Website|E-?mail|Registered Office|Scrip Code|CIN|Page|Date|Time|Q\d|FY\d*)$", re.I)
PAGE = re.compile(r"^\s*Page \d+ of \d+\s*$", re.I | re.M)
QA_START = re.compile(r"(?:first|1st|next|following)\s+question|question[- ]and[- ]answer|open the (?:floor|line)s?|begin the q\s*&\s*a", re.I)
ANALYST_INTRO = re.compile(
    r"question\s+(?:is|comes|will be)?\s*(?:from|is from)\s+(?:the line of\s+)?(?:Mr\.?\s+|Ms\.?\s+|Mrs\.?\s+)?"
    r"(?P<name>[A-Z][A-Za-z\.\'’\-]+(?:\s+[A-Z][A-Za-z\.\'’\-]+){0,3})", re.I)
CALL_TIME = re.compile(r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\s*,?\s*(\d{1,2}[:.]\d{2})\s*hrs\s*IST", re.I)
CALL_DATE = re.compile(r"^\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\s*$", re.M)
CALL_HEADING = re.compile(r"^.*earnings\s+(?:conference\s+)?call.*$", re.I | re.M)
NAME_LINE = re.compile(r"^(?:(?:Mr|Ms|Mrs|Dr)\.?\s+)?[A-Z][A-Za-z\.\'’\-]+(?:\s+[A-Z][A-Za-z\.\'’\-]+){0,3}$")
ANALYST_LIST = re.compile(r"^\s*(?:ANALYSTS?|PARTICIPANTS? FROM THE ANALYST COMMUNITY)\s*:?\s*$", re.M)
MGMT_LIST = re.compile(r"^\s*(?:CORPORATE PARTICIPANTS|MANAGEMENT)\s*:?\s*$", re.M)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])")

HEDGE = re.compile(r"\b(?:may|might|could|possibl[ey]|perhaps|uncertain(?:ty|ties)?|unclear|unpredictab\w*|cautious(?:ly)?|"
                   r"challeng(?:e|es|ing)|headwinds?|volatil\w*|soft(?:ness|er)?|slow(?:down|ing|er)|pressures?|delay(?:s|ed)?|"
                   r"difficult|risks?|depend(?:s|ing)? on|not sure|hard to say|too early)\b", re.I)
FORWARD = re.compile(r"\b(?:expect(?:s|ed|ing)?|outlook|guidance|going forward|next (?:quarter|year)|coming (?:quarters?|year)|"
                     r"pipeline|anticipat\w*|visibility|FY\s?\d{2,4}|we (?:see|believe|aim|plan|will)|should (?:see|be))\b", re.I)
NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|per ?cent|bps|basis points|billion|million|crore)", re.I)
PLEASANTRY = re.compile(r"^(?:thank you|thanks|good (?:morning|afternoon|evening|day)|hi|hello|yes|sure|okay|ok)\b[^.]{0,40}[.!]?$", re.I)


def call_input(sentence: str, target: str | None, context: str, fmt: dict | None = None) -> tuple[str, str]:
    """Model input for the call model: (company + speaker role + section, sentence with markers).
    context is e.g. 'management prepared', 'management answer', 'analyst question'.
    Markers: <t>company</t>, <o>peers</o>, <c>/<a> contrast words, <m> market indices (as for news), plus <h>…</h>
    around hedging words — the one marker that is call-specific."""
    from entity import model_input
    f = fmt or {}
    _, b = model_input(sentence, target or "__none__", entity_markers=f.get("t", True) and bool(target),
                       other_markers=f.get("o", True), contrast_tags=f.get("c", True), market_tags=f.get("m", True))
    if f.get("h", True):
        b = HEDGE.sub(lambda m: f"<h> {m.group(0)} </h>", b)
    return f"{COMPANY.get(target, 'the company')} {context}", b


@dataclass
class ParsedCall:
    turns: pd.DataFrame        # speaker, role, section, text
    sentences: pd.DataFrame    # speaker, role, section, context, sentence, hedge, forward, number
    call_time: pd.Timestamp | None
    analysts: list[str]


class TranscriptParser:
    """Text of a BSE transcript PDF → speaker turns → sentences with role and section."""

    def __init__(self, min_words: int = 6):
        self.min_words = min_words

    def _body(self, text: str) -> str:
        text = PAGE.sub("", text.replace("\r", ""))
        start = 0
        for h in CALL_HEADING.finditer(text):                  # Infosys files put the press conference first
            nxt = START.search(text, h.end())
            if nxt and nxt.start() - h.end() < 6000: start = h.start()
        m = START.search(text, start)
        return text[m.start():] if m else text[start:]

    @staticmethod
    def _listed(text: str, heading: re.Pattern) -> list[str]:
        """Names under a participants heading (Infosys layout: name line, title/firm line, name line, …)."""
        m = heading.search(text)
        if not m: return []
        lines = [l.strip() for l in text[m.end():m.end() + 3000].split("\n") if l.strip()]
        out = []
        for i, l in enumerate(lines):
            if START.match(l) or "©" in l or l.isdigit() or ANALYST_LIST.match(l) or MGMT_LIST.match(l): break
            if i % 2 == 0 and NAME_LINE.match(l): out.append(l)
        return out

    def _listed_analysts(self, text: str) -> set[str]:
        return {self._norm(n) for n in self._listed(text, ANALYST_LIST)}

    @staticmethod
    def _name_lines(body: str) -> set[str]:
        """Speaker names written alone on a line (Infosys layout): a name-shaped line seen at least twice, or 'Moderator'."""
        cand = [l.strip() for l in body.split("\n") if NAME_LINE.match(l.strip()) and len(l.split()) <= 4]
        counts = pd.Series(cand).value_counts() if cand else pd.Series(dtype=int)
        return {n for n, c in counts.items() if c >= 2 or re.fullmatch(r"(?i)moderator|operator", n)}

    def turns(self, text: str) -> pd.DataFrame:
        rows, cur = [], None
        body = self._body(text)
        colon_style = sum(bool(SPEAKER.match(l.strip())) for l in body.split("\n")) >= 10
        alone = set() if colon_style else (self._name_lines(body) | set(self._listed(text, ANALYST_LIST)) | set(self._listed(text, MGMT_LIST)))
        for line in body.split("\n"):
            line = line.strip()
            if not line or "©" in line or line.isdigit(): continue
            m = SPEAKER.match(line) if colon_style else None
            if m and not NOT_SPEAKER.match(m.group("name").strip()) and len(m.group("name").split()) <= 5:
                if cur: rows.append(cur)
                cur = {"speaker": m.group("name").strip(), "text": m.group("rest").strip()}
            elif line in alone:
                if cur: rows.append(cur)
                cur = {"speaker": line, "text": ""}
            elif cur:
                cur["text"] = (cur["text"] + " " + line).strip()
        if cur: rows.append(cur)
        df = pd.DataFrame(rows, columns=["speaker", "text"])
        if df.empty: return df.assign(role=[], section=[])
        mod = df["speaker"].str.fullmatch(r"(?i)moderator|operator|coordinator|host")
        analysts = {self._norm(n) for t in df.loc[mod, "text"] for n in ANALYST_INTRO.findall(t)} | self._listed_analysts(text)
        df["role"] = np.where(mod, "moderator",
                              np.where(df["speaker"].map(self._norm).isin(analysts) | self._analyst_by_order(df, mod), "analyst", "management"))
        qa_idx = next((i for i, (is_mod, t) in enumerate(zip(mod, df["text"])) if is_mod and QA_START.search(t)), len(df))
        df["section"] = np.where(np.arange(len(df)) < qa_idx, "prepared", "qa")
        df.attrs["analysts"] = sorted(analysts)
        return df

    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"^(?:mr|ms|mrs|dr)\.?\s+", "", name.strip().lower())

    @staticmethod
    def _analyst_by_order(df: pd.DataFrame, mod: pd.Series) -> pd.Series:
        """Fallback: in Q&A, the first speaker after a moderator turn that announces a question is the analyst."""
        flag = pd.Series(False, index=df.index)
        for i in df.index[mod]:
            if ANALYST_INTRO.search(df.at[i, "text"]) and i + 1 in df.index and not mod.get(i + 1, False):
                flag.at[i + 1] = True
        names = set(df.loc[flag, "speaker"])
        return df["speaker"].isin(names)

    def parse(self, text: str) -> ParsedCall:
        t = self.turns(text)
        rows = []
        for r in t.itertuples():
            if r.role == "moderator": continue
            ctx = ("analyst question" if r.role == "analyst" else
                   "management answer" if r.section == "qa" else "management prepared")
            for s in SENT_SPLIT.split(r.text):
                s = re.sub(r"\s+", " ", s).strip()
                if len(s.split()) < self.min_words or PLEASANTRY.match(s): continue
                rows.append({"speaker": r.speaker, "role": r.role, "section": r.section, "context": ctx, "sentence": s,
                             "hedge": bool(HEDGE.search(s)), "forward": bool(FORWARD.search(s)), "number": bool(NUMBER.search(s))})
        sents = pd.DataFrame(rows, columns=["speaker", "role", "section", "context", "sentence", "hedge", "forward", "number"])
        head = text[:6000]
        m = CALL_TIME.search(head)
        if m:
            when = pd.to_datetime(f"{m.group(1)} {m.group(2).replace('.', ':')}", errors="coerce")
        else:                                                    # date only (Wipro, Infosys): the last date line before the call opens
            body_at = text.find(self._body(text)[:200]); dates = CALL_DATE.findall(text[:max(body_at, 0)] or head)
            when = pd.to_datetime(dates[-1].replace(",", ""), errors="coerce") if dates else None
        return ParsedCall(t, sents, when, t.attrs.get("analysts", []))


class CallFeatures:
    """Sentence sentiment (column `s` = P(pos) − P(neg)) → one row of call-level features."""

    NEG = -0.5

    @staticmethod
    def compute(sents: pd.DataFrame) -> dict:
        if sents.empty: return {}
        s = sents
        prep = s[s.context == "management prepared"]; ans = s[s.context == "management answer"]; q = s[s.context == "analyst question"]
        qa = s[s.section == "qa"]; mgmt = s[s.role == "management"]; fwd = mgmt[mgmt.forward]
        mean = lambda d, c="s": float(d[c].mean()) if len(d) else np.nan
        return {
            "n_sentences": int(len(s)), "qa_share": float(len(qa) / len(s)), "n_analyst_sentences": int(len(q)),
            "tone_all": mean(s), "tone_prepared": mean(prep), "tone_answers": mean(ans), "tone_analysts": mean(q),
            "tone_gap_qa_minus_prepared": mean(ans) - mean(prep) if len(ans) and len(prep) else np.nan,
            "neg_share_qa": float((qa.s < CallFeatures.NEG).mean()) if len(qa) else np.nan,
            "neg_share_analysts": float((q.s < CallFeatures.NEG).mean()) if len(q) else np.nan,
            "hedge_rate_answers": mean(ans, "hedge"), "hedge_rate_prepared": mean(prep, "hedge"),
            "forward_share": mean(mgmt, "forward"), "forward_tone": mean(fwd),
            "dispersion_qa": float(qa.s.std()) if len(qa) > 1 else np.nan,
            "magnitude_qa": float(qa.s.abs().mean()) if len(qa) else np.nan,
        }
