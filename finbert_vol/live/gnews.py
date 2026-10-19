"""Headlines from Google News RSS search, restricted to a few established Indian business news sites.

Why the RSS search feed and not scraping google.com results pages: the feed is Google's public, keyless interface for
news search, returns structured items (headline, publisher, publication time), supports the after:/before: date
operators, and is a few kilobytes per request. Only headlines are kept — no article pages are fetched.

One query per date window:
    ("Tata Consultancy" OR TCS) (site:economictimes.indiatimes.com OR site:moneycontrol.com OR …) after:A before:B
A feed returns at most ~100 items, so a window that comes back (nearly) full is split in half until it isn't
(down to one day). Polite: one request every 1–2.5 s, exponential back-off on 429 / 5xx.
"""
from __future__ import annotations
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd

FEED = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
SITES = {  # domain → short source name (the four sources of the HF training corpus + Mint)
    "economictimes.indiatimes.com": "economictimes", "moneycontrol.com": "moneycontrol",
    "business-standard.com": "businessstandard", "financialexpress.com": "financialexpress", "livemint.com": "livemint",
}
QUERY_NAMES = {"TCS": ['"Tata Consultancy"', "TCS"], "INFY": ["Infosys"], "WIPRO": ["Wipro"],
               "HCLTECH": ['"HCL Tech"', "HCLTech"], "TECHM": ['"Tech Mahindra"']}
FULL = 95            # items at which a window is treated as truncated
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def query(target: str, sites=tuple(SITES), start: date | None = None, end: date | None = None) -> str:
    names = " OR ".join(QUERY_NAMES.get(target, [target]))
    q = f"({names}) (" + " OR ".join(f"site:{s}" for s in sites) + ")"
    if start: q += f" after:{start:%Y-%m-%d}"
    if end: q += f" before:{end:%Y-%m-%d}"          # Google's before: is exclusive
    return q


def parse(xml_text: str) -> pd.DataFrame:
    """RSS → one row per item: published (UTC), date (IST calendar day), title (publisher suffix removed), source, url."""
    rows = []
    root = ET.fromstring(xml_text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source"); src_name = (src_el.text or "").strip() if src_el is not None else ""
        src_url = src_el.get("url", "") if src_el is not None else ""
        if src_name and title.endswith(f" - {src_name}"): title = title[: -len(src_name) - 3].strip()
        try: pub = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate") or "")).tz_convert("UTC")
        except Exception: continue
        dom = next((d for d in SITES if d in src_url), "")
        rows.append({"published": pub, "date": pub.tz_convert("Asia/Kolkata").tz_localize(None).normalize(), "title": title,
                     "source": SITES.get(dom, src_name.lower().replace(" ", "")), "url": it.findtext("link") or ""})
    return pd.DataFrame(rows, columns=["published", "date", "title", "source", "url"])


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


class GoogleNewsRSS:
    def __init__(self, target: str = "TCS", sites=tuple(SITES), delay: float = 1.0, session=None):
        import requests
        self.target, self.sites, self.delay = target, tuple(sites), delay
        self.s = session or requests.Session(); self.n_requests = 0

    def _get(self, url: str) -> str:
        for attempt in range(5):
            try:
                r = self.s.get(url, headers={"User-Agent": UA, "Accept": "application/rss+xml,application/xml"}, timeout=30)
                self.n_requests += 1
                if r.status_code == 200: return r.text
                if r.status_code not in (429, 500, 502, 503): r.raise_for_status()
                err = f"HTTP {r.status_code}"
            except Exception as e: err = str(e)
            wait = min(120, 2 ** (attempt + 2)) + random.uniform(0, 3)
            print(f"  google news: {err} — retry in {wait:.0f}s", flush=True); time.sleep(wait)
        raise RuntimeError(f"Google News failed: {url}")

    def window(self, start: date, end: date) -> pd.DataFrame:
        """Headlines published in [start, end) (IST days). Splits windows the feed truncates."""
        df = parse(self._get(FEED.format(q=quote_plus(query(self.target, self.sites, start, end)))))
        time.sleep(random.uniform(1.0, 2.5) * self.delay)
        if len(df) >= FULL and (end - start).days > 1:
            mid = start + timedelta(days=(end - start).days // 2)
            return pd.concat([self.window(start, mid), self.window(mid, end)], ignore_index=True)
        return df

    def fetch(self, start: date, end: date, step_days: int = 7, log=print) -> pd.DataFrame:
        """All headlines in [start, end], in step_days windows; de-duplicated by normalised title."""
        out, a = [], start
        while a <= end:
            b = min(a + timedelta(days=step_days), end + timedelta(days=1))
            w = self.window(a, b); out.append(w)
            if log and (a.day <= step_days or b > end): log(f"  {a} → {b - timedelta(days=1)}: {len(w)} headlines")
            a = b
        df = pd.concat(out, ignore_index=True) if out else parse("<rss/>")
        return dedupe(df)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Same headline from the same day (syndication, re-crawls): keep the earliest."""
    if df.empty: return df
    df = df.assign(_k=df["title"].map(norm_title)).sort_values("published")
    return df.drop_duplicates(["_k", "date"]).drop(columns="_k").reset_index(drop=True)
