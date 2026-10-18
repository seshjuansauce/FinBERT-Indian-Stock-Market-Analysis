"""Unit tests for finbert_vol — no GPU, no downloads: a DummyModel stands in for FinBERT.

    python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

import finbert_vol as fv
from finbert_vol.calls import CallFeatures, TranscriptParser, call_input
from finbert_vol.features import CALL_FEATURES, FeatureBuilder
from finbert_vol.models import LABELS, SentimentModel
from finbert_vol.scorers import AnnouncementScorer, CallScorer, NewsScorer
from finbert_vol.volatility import ATT, HAR, Evaluator, VolatilityModel

TRANSCRIPT = """TCS/SE/10/2026-27
Dear Sirs,
Sub: Transcript of the earnings conference call
Tata Consultancy Services Limited
Financial Results Q4 & Full Year FY 2026 Conference Call
April 09, 2026,19:00 hrs IST (08:30 hrs US ET)
Moderator: Ladies and gentlemen, good day and welcome to the TCS Earnings
Conference Call. I now hand the conference over to Ms. Nehal Shah.
Nehal Shah: Thank you, operator. Good evening and welcome everyone to our results call today.
K Krithivasan: We delivered strong sequential growth of 1.2% in constant currency this quarter.
Margins expanded despite wage hikes and we expect the pipeline to stay healthy next quarter.
Page 1 of 34
Samir Seksaria: Operating margin was 25.2%, up 80 basis points sequentially, driven by productivity.
Moderator: Thank you. We will now begin the question-and-answer session. The first question is from the line of Kumar Rakesh from BNP Paribas.
Kumar Rakesh: Hi, thanks for the opportunity. Why did BFSI growth slow down so sharply this quarter?
K Krithivasan: Some clients may delay discretionary spending given the uncertainty, but the deal pipeline is still strong.
Moderator: Thank you. The next question is from the line of Ankur Rudra from JP Morgan.
Ankur Rudra: Can you comment on the headwinds in Europe and whether margins could come under pressure?
Samir Seksaria: It is too early to say, but we remain confident of our long-term margin band of 26 to 28%.
"""


class DummyModel(SentimentModel):
    """Positive if the text says strong/healthy/expanded/confident, negative if slow/headwinds/pressure/delay, else neutral."""

    def __init__(self, fmt=None, T=1.0):
        self.path, self.temperature, self.batch_size, self.max_len, self.name = "dummy", T, 64, 128, "dummy"
        self.fmt = fmt or {"format": "pair_tocm", "pair": True, "t": True, "o": True, "c": True, "m": True}
        self.labels = list(LABELS)

    def logits(self, texts, targets=None, contexts=None):
        self.last_inputs = self.format_inputs(texts, targets, contexts)
        z = []
        for t in texts:
            t = t.lower(); pos = sum(w in t for w in ("strong", "healthy", "expanded", "confident"))
            neg = sum(w in t for w in ("slow", "headwind", "pressure", "delay"))
            z.append([2.0 * pos, 2.0 * neg, 1.0])
        return np.array(z, dtype=float)


# ---------------- transcript parsing ----------------
def test_parser_roles_sections_and_time():
    pc = TranscriptParser().parse(TRANSCRIPT)
    t = pc.turns
    assert set(t.role) == {"moderator", "management", "analyst"}
    assert set(t[t.role == "analyst"].speaker) == {"Kumar Rakesh", "Ankur Rudra"}
    assert (t[t.speaker == "K Krithivasan"].section.tolist()) == ["prepared", "qa"]
    assert pc.call_time == pd.Timestamp("2026-04-09 19:00")
    assert not pc.sentences.sentence.str.contains("Page 1 of").any()          # page footer removed
    ctx = set(pc.sentences.context)
    assert ctx == {"management prepared", "management answer", "analyst question"}


def test_call_specific_lexicons():
    pc = TranscriptParser().parse(TRANSCRIPT); s = pc.sentences
    hedged = s[s.hedge].sentence.str.cat(sep=" ").lower()
    assert "may delay" in hedged and "too early to say" in hedged
    assert s[s.sentence.str.contains("pipeline to stay healthy")].forward.all()
    assert s[s.sentence.str.contains("25.2%")].number.all()


def test_call_input_markers():
    a, b = call_input("Some clients may delay spending but TCS margins expanded, unlike Infosys", "TCS", "management answer",
                      {"t": True, "o": True, "c": True, "m": True, "h": True})
    assert a == "tcs management answer"
    assert "<t> TCS </t>" in b and "<o> Infosys </o>" in b and "<h> may </h>" in b and "<a> but </a>" in b


# ---------------- models / scorers ----------------
def test_probs_and_predict():
    m = DummyModel(T=2.0)
    df = m.predict(["TCS growth strong", "Headwinds and pressure", "TCS board meets"], "TCS")
    assert list(df.label) == ["positive", "negative", "neutral"]
    np.testing.assert_allclose(df[[f"p_{l}" for l in LABELS]].sum(axis=1), 1.0)
    hot = SentimentModel.probs(np.array([[4.0, 0, 0]]), 1.0)[0, 0]; cool = SentimentModel.probs(np.array([[4.0, 0, 0]]), 2.0)[0, 0]
    assert cool < hot                                            # temperature > 1 softens confidence
    assert m.last_inputs[0][0] == "tcs" and "<t> TCS </t>" in m.last_inputs[0][1]


def test_news_scorer_adds_columns():
    art = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "title": ["TCS posts strong Q3", "TCS faces headwinds"],
                        "focus_text": ["", "TCS: deal pipeline healthy and strong this quarter overall"]})
    out = NewsScorer(DummyModel(), "TCS").score(art, tag="C")
    assert out["sent_C"].iloc[0] > 0 > out["sent_C"].iloc[1]
    assert np.isnan(out["sent_C_focus"].iloc[0]) and out["sent_C_focus"].iloc[1] > 0


def test_call_scorer_features():
    m = DummyModel(fmt={"format": "call", "pair": True, "t": True, "o": True, "c": True, "m": True, "h": True})
    sents, feats = CallScorer(m, transcripts=None).score_one(TRANSCRIPT, "TCS")
    assert feats["n_analysts"] == 2 and feats["call_time"] == pd.Timestamp("2026-04-09 19:00")
    assert feats["tone_analysts"] < 0 < feats["tone_prepared"]            # sceptical questions, upbeat script
    assert 0 < feats["hedge_rate_answers"] <= 1 and feats["qa_share"] > 0
    assert m.last_inputs[0][0].startswith("tcs management prepared")


# ---------------- features: timing / no look-ahead ----------------
def test_call_daily_is_causal():
    td = pd.bdate_range("2024-01-01", periods=20)
    calls = pd.DataFrame({"published": [pd.Timestamp("2024-01-06")], **{f: [0.5] for f in CALL_FEATURES}})  # a Saturday
    d = FeatureBuilder.call_daily(calls, td, horizon=3)
    live = d.index[d.call_live == 1].tolist()
    assert live == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-09")]  # Sat → Fri row
    assert d.loc[:"2024-01-04", "call_live"].sum() == 0 and d.loc["2024-01-08", "days_since_call"] == 1


def test_announcement_flags(tmp_path):
    f = pd.DataFrame({
        "isin": ["INE467B01029"] * 4,
        "date": pd.to_datetime(["2024-01-02", "2024-01-11", "2024-01-15", "2024-01-03"]),
        "announced_at": pd.NaT, "after_close": False, "bse_category": "",
        "bse_subcategory": ["Analyst / Investor Meet", "Financial Results", "Earnings Call Transcript", "General"],
        "bse_headline": ["Intimation of earnings conference call on Jan 11", "Results", "Transcript", "x"],
        "path": ["", "", "pdfs/x.pdf", ""]})
    p = tmp_path / "filings.parquet"; f.to_parquet(p)
    flags = AnnouncementScorer(fv.BSEAnnouncements(p)).daily_flags("INE467B01029", pd.bdate_range("2024-01-01", "2024-01-31"))
    assert flags.loc["2024-01-02":"2024-01-11", "call_upcoming"].eq(1).all() and flags.loc["2024-01-12", "call_upcoming"] == 0
    assert flags.loc["2024-01-11", "results_filing"] == 1 and flags.loc["2024-01-15", "transcript_published"] == 1


# ---------------- volatility ----------------
def _synthetic_table(n=400, seed=0):
    rng = np.random.default_rng(seed); idx = pd.bdate_range("2022-01-03", periods=n)
    d = pd.DataFrame(index=idx)
    for c in HAR: d[c] = rng.normal(0, 1, n)
    for c in ATT: d[c] = rng.normal(0, 1, n)
    d["gjr_fc"] = np.exp(rng.normal(0.5, 0.2, n)); d["u_gjr"] = 0.6 * d["log_n"] + rng.normal(0, 0.5, n)
    d["rv_target"] = d["gjr_fc"] * np.exp(d["u_gjr"]); return d


def test_walk_forward_learns_signal_and_latest():
    d = _synthetic_table()
    fc = VolatilityModel({"base": HAR, "news": HAR + ATT}, start="2022-09-01").forecasts(d)
    ev = Evaluator(("2022-09-01", "2022-12-31"), "2023-01-01").table(fc, baseline="base")
    t = ev[(ev.period == "test")].set_index("model")
    assert t.loc["news", "qlike"] < t.loc["base", "qlike"] and t.loc["news", "p"] < 0.05
    out = VolatilityModel({}).latest(d, HAR + ATT)
    assert out["sigma_pct"] > 0 and out["as_of"] == str(d.index[-1].date())


def test_call_window_test():
    rng = np.random.default_rng(1); n = 60
    panel = pd.DataFrame({"company": rng.choice(["TCS", "INFY", "WIPRO"], n), "hedge_rate_answers": rng.normal(0, 1, n),
                          "tone_answers": rng.normal(0, 1, n)})
    panel["u_5d"] = 0.8 * panel["hedge_rate_answers"] + rng.normal(0, 0.3, n)
    res = Evaluator.call_window_test(panel, ["hedge_rate_answers", "tone_answers"], "u_5d").set_index("feature")
    assert res.loc["hedge_rate_answers", "p"] < 0.001 and res.loc["tone_answers", "p"] > 0.01


INFOSYS_STYLE = """Rishi Basu
Thank you. With that we come to the end of this press conference. Please join us for high tea outside.
Infosys Limited
Q4 FY26 Earnings Conference Call
April 23, 2026
CORPORATE PARTICIPANTS:
Salil Parekh
Chief Executive Officer and Managing Director
ANALYSTS
Yogesh Aggarwal
HSBC Securities
Ankur Rudra
JP Morgan
1
Moderator
Ladies and gentlemen, greetings, and welcome to Infosys Limited Q4 FY26 Earnings Conference Call.
Salil Parekh
We delivered a strong performance in the financial year with healthy deal wins across all segments.
Moderator
Thank you. The first question comes from the line of Yogesh Aggarwal from HSBC. Please go ahead.
Yogesh Aggarwal
Why has the growth guidance for next year been set so cautiously given the strong deal wins?
Salil Parekh
The macro environment remains uncertain and some clients may delay decisions in the near term.
Moderator
The next question comes from the line of Ankur Rudra from JP Morgan.
Ankur Rudra
Could you talk about pricing pressure in the financial services vertical this quarter?
Salil Parekh
We see some pricing pressure but our margins remain within the guided band for the year.
"""


def test_parser_infosys_layout_skips_press_conference():
    pc = TranscriptParser().parse(INFOSYS_STYLE)
    t = pc.turns
    assert "Rishi Basu" not in set(t.speaker)                                  # press conference dropped
    assert set(t[t.role == "analyst"].speaker) == {"Yogesh Aggarwal", "Ankur Rudra"}
    assert set(t[t.role == "management"].speaker) == {"Salil Parekh"}
    assert t[t.speaker == "Salil Parekh"].section.tolist() == ["prepared", "qa", "qa"]
    assert pc.call_time == pd.Timestamp("2026-04-23")
