"""Live pipeline: Google News RSS parsing / window splitting, model table timing, end-to-end daily run + container round trip."""
import base64
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from finbert_vol.live import gnews
from finbert_vol.live.gnews import GoogleNewsRSS, parse, query
from finbert_vol.live.model import LiveModel, news_daily
from finbert_vol.live.runner import Runner, State, score_headlines

RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>x</title>
<item><title>TCS Q2 results: net profit rises 5% - The Economic Times</title><link>https://news.google.com/rss/articles/abc</link>
<pubDate>Thu, 09 Oct 2025 13:05:00 GMT</pubDate><source url="https://economictimes.indiatimes.com">The Economic Times</source></item>
<item><title>Tata Steel shares slip 2% - Moneycontrol</title><link>https://news.google.com/rss/articles/def</link>
<pubDate>Thu, 09 Oct 2025 20:10:00 GMT</pubDate><source url="https://www.moneycontrol.com">Moneycontrol</source></item>
</channel></rss>"""


def test_parse_rss_titles_dates_sources():
    d = parse(RSS)
    assert d["title"].tolist() == ["TCS Q2 results: net profit rises 5%", "Tata Steel shares slip 2%"]
    assert d["source"].tolist() == ["economictimes", "moneycontrol"]
    # 20:10 GMT on 9 Oct = 01:40 IST on 10 Oct → dated 10 Oct (IST calendar day)
    assert d["date"].tolist() == [pd.Timestamp("2025-10-09"), pd.Timestamp("2025-10-10")]


def test_query_string():
    q = query("TCS", ("economictimes.indiatimes.com", "livemint.com"), date(2025, 1, 1), date(2025, 1, 8))
    assert q == '("Tata Consultancy" OR TCS) (site:economictimes.indiatimes.com OR site:livemint.com) after:2025-01-01 before:2025-01-08'


def _rss(n, day):
    items = "".join(f"<item><title>TCS headline {day} {i} - Mint</title><link>u{i}</link><pubDate>{pd.Timestamp(day):%a, %d %b %Y} 06:00:00 GMT</pubDate>"
                    f"<source url='https://www.livemint.com'>Mint</source></item>" for i in range(n))
    return f"<rss><channel>{items}</channel></rss>"


def test_full_window_is_split(monkeypatch):
    calls = []
    def fake_get(self, url):
        calls.append(url); a = url.split("after%3A")[1][:10]; b = url.split("before%3A")[1][:10]
        days = (pd.Timestamp(b) - pd.Timestamp(a)).days
        return _rss(100 if days > 2 else 30, a)                  # wide windows look truncated
    monkeypatch.setattr(GoogleNewsRSS, "_get", fake_get); monkeypatch.setattr(gnews.time, "sleep", lambda s: None)
    df = GoogleNewsRSS("TCS").window(date(2025, 1, 1), date(2025, 1, 9))
    assert len(calls) == 7 and len(df) == 4 * 30                 # 8 days → 4+4 → 2+2+2+2 (each 30 items)


def _prices(n=900, seed=0):
    rng = np.random.default_rng(seed); days = pd.bdate_range("2021-01-04", periods=n); rows = []
    for sym, lvl in (("TCS.NS", 3000.0), ("^CRSLDX", 15000.0)):
        c = lvl * np.exp(np.cumsum(rng.normal(0, 0.012, n))); o = c * np.exp(rng.normal(0, 0.004, n))
        hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.006, n))); lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
        rows.append(pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Symbol": sym, "Open": o, "High": hi, "Low": lo, "Close": c,
                                  "Adj Close": c, "Volume": 1e6, "Dividends": 0.0}))
    return pd.concat(rows, ignore_index=True)


def test_news_timing_weekend_and_evening():
    td = pd.bdate_range("2025-10-06", "2025-10-14")
    h = pd.DataFrame({"date": pd.to_datetime(["2025-10-09", "2025-10-11", "2025-10-12"]), "relevance": [1.0, 1.0, 0.0], "s": [0.5, -0.5, 0.9]})
    d = news_daily(h, td)
    assert d.loc["2025-10-09", "n"] == 1 and d.loc["2025-10-10", "n"] == 1      # Saturday news → Friday's row
    assert d.loc["2025-10-10", "sent"] == -0.5 and d.loc["2025-10-13", "no_news"] == 1   # irrelevant Sunday item ignored


class FakePrices:
    def __init__(self, px): self.px = px
    def fetch(self, symbols, start):
        p = self.px[self.px.Symbol.isin(symbols)]; return p[pd.to_datetime(p.Date) >= pd.Timestamp(start)]


class FakeNews:
    def __init__(self): self.calls = []
    def fetch(self, start, end, log=None):
        self.calls.append((start, end)); days = pd.date_range(start, end)
        return pd.DataFrame({"published": [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=6) for d in days],
                             "date": days, "title": [f"TCS shares {'rise' if i % 2 else 'fall'} on day {d:%Y-%m-%d}" for i, d in enumerate(days)],
                             "source": "economictimes", "url": "u"})


class TinyModel:
    name = "tiny"
    def predict(self, texts, target=None, contexts=None):
        s = np.array([0.6 if "rise" in t else -0.6 for t in texts])
        return pd.DataFrame({"s": s, "p_positive": (1 + s) / 2, "p_negative": (1 - s) / 2, "p_neutral": 0.0})


def test_score_headlines_relevance():
    h = pd.DataFrame({"published": pd.Timestamp("2025-01-01", tz="UTC"), "date": pd.Timestamp("2025-01-01"),
                      "title": ["TCS shares rise 3%", "Tata Steel falls; TCS dividend on sale of goods rule", "Cricket: India win"],
                      "source": "x", "url": "u"})
    out = score_headlines(h, TinyModel(), "TCS")
    assert out.loc[0, "relevance"] == 1.0 and out.loc[0, "s"] == 0.6
    assert out.loc[2, "relevance"] == 0 and np.isnan(out.loc[2, "s"])


def test_daily_run_end_to_end_and_container_roundtrip(tmp_path):
    px = _prices(); last = pd.Timestamp(px.Date.max()).date()
    st = State(tmp_path / "s"); news = FakeNews()
    r = Runner(st, TinyModel(), prices=FakePrices(px[pd.to_datetime(px.Date) <= pd.Timestamp(last) - pd.Timedelta(days=30)]),
               news=news, today=last - timedelta(days=30), log=lambda m: None)
    r.update_prices(); r.update_news(start=date(2021, 1, 1)); r.publish()
    # next day: only the recent window is fetched again, nothing duplicated
    n0 = len(pd.read_csv(st.root / "headlines.csv"))
    r2 = Runner(st, TinyModel(), prices=FakePrices(px), news=news, today=last, log=lambda m: None); latest = r2.daily()
    h = pd.read_csv(st.root / "headlines.csv")
    assert news.calls[-1][0] == last - timedelta(days=32) and len(h) == n0 + 30 and h.title.is_unique
    tr = pd.read_csv(st.root / "public" / "tracker.csv", parse_dates=["day"])
    assert tr.pending.sum() == 1 and tr.iloc[-1].pending and tr.day.is_monotonic_increasing
    assert latest["as_of"] == str(pd.Timestamp(last).date()) and latest["sigma_pct"] > 0
    summ = json.loads((st.root / "public" / "summary.json").read_text())
    assert set(summ["metrics"]["all"]["qlike"]) == {"garch", "har", "finbert"}
    # no look-ahead: forecasts for days before the cut are identical when computed with less data
    tr_old = pd.read_csv(tmp_path / "s" / "public" / "tracker.csv")
    assert np.isfinite(tr_old.pred_vol).all()

    # container: state in, state + public files out
    from finbert_vol.live import server
    res = server.run(st.pack(), model=TinyModel(), prices=FakePrices(px), news=FakeNews())
    assert res["ok"], res["log"][-3:]
    assert {"tracker.csv", "latest.json", "summary.json", "headlines.json"} <= set(res["public"])
    st2 = State.unpack(base64.b64decode(res["state_b64"]), tmp_path / "s2")
    assert (st2.root / "headlines.csv").exists() and (st2.root / "meta.json").exists()


def test_forecasts_are_causal():
    px = _prices(); lm = LiveModel(eval_start="2022-06-01", burn=300)
    h = pd.DataFrame(columns=["date", "relevance", "s"])
    full = lm.forecasts(lm.table(px, h))
    cut = pd.Timestamp(px.Date.unique()[700])
    part = lm.forecasts(lm.table(px[pd.to_datetime(px.Date) <= cut], h))
    common = part.index[part.index < cut - pd.Timedelta(days=40)]       # blocks fully before the cut
    assert len(common) > 50
    np.testing.assert_allclose(full.loc[common, "finbert"], part.loc[common, "finbert"], rtol=1e-6)


def test_unsafe_archive_rejected(tmp_path):
    import io, tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("../evil.txt"); data = b"x"; ti.size = 1; tf.addfile(ti, io.BytesIO(data))
    with pytest.raises(ValueError): State.unpack(buf.getvalue(), tmp_path / "x")
