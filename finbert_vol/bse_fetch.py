"""Earnings-call transcripts for the NIFTY 50 from BSE corporate announcements (run on your Mac — BSE blocks cloud IPs).

  python -m finbert_vol fetch-transcripts                       # all of NIFTY 50, 2021-01-01 → today
  python -m finbert_vol fetch-transcripts --symbols CIPLA,DRREDDY --start 2021-01-01

Writes
  data/calls/bse_filings.parquet    every transcript filing found: symbol, isin, date, announced_at, headline, path …
  data/calls/pdfs/<SYMBOL>/<YYYYMMDD>_<newsid>.pdf
  data/calls/transcripts.jsonl      extracted text, appended (key = path, same cache the scorer reads)
Resumable: (symbol, year) windows already fetched are recorded in data/calls/fetch_state.json and skipped.

Politeness: one browser-like session seeded from bseindia.com, 0.6–2 s jittered delay between pages, backoff on
403/429 — the same pattern as axiom's BSE ingestion.
"""
from __future__ import annotations
import json, random, re, time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .universe import BSE_CODE, NAME_CHECK

API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
SEED = "https://www.bseindia.com/corporates/ann.html"
PDF_URLS = ("https://www.bseindia.com/xml-data/corpfiling/AttachHis/{}", "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{}")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

# A transcript filing: BSE's own subcategory (from ~2022), or — before that — a Company Update whose headline says it is
# the transcript of an earnings / results / analyst call. AGMs, investor days and healthcare conferences are excluded.
TRANSCRIPT = re.compile(r"transcript", re.I)
CALL = re.compile(r"earnings|results?|quarter|q[1-4]\b|conference call|con-?call|analysts?|investors?\s+call|press conference", re.I)
NOT_CALL = re.compile(r"annual general|\bagm\b|healthcare conference|investor day|analyst day|capital markets day|j\.?p\.? morgan", re.I)


def is_transcript(subcategory: str, headline: str) -> bool:
    if subcategory == "Earnings Call Transcript": return True
    h = headline or ""
    return bool(TRANSCRIPT.search(h) and CALL.search(h) and not NOT_CALL.search(h))


class BSETranscriptFetcher:
    def __init__(self, root: Path, delay: float = 1.0, category: str = "Company Update"):
        import requests
        self.root, self.delay, self.category = Path(root), delay, category
        self.dir = self.root / "data" / "calls"; (self.dir / "pdfs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "fetch_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"done": []}
        self.s = requests.Session()
        self.h = {"User-Agent": UA, "Referer": SEED, "Origin": "https://www.bseindia.com", "Accept": "application/json, text/plain, */*",
                  "Accept-Language": "en-IN,en;q=0.9"}
        self._seed()

    def _seed(self):
        try: self.s.get(SEED, headers={**self.h, "Accept": "text/html,*/*"}, timeout=20); time.sleep(random.uniform(1, 2))
        except Exception as e: print(f"  session seed failed ({e}); continuing")

    def _sleep(self, k=1.0): time.sleep(random.uniform(0.6, 2.0) * self.delay * k)

    def _get(self, params: dict) -> list[dict]:
        for attempt in range(4):
            try:
                r = self.s.get(API, params=params, headers=self.h, timeout=30, allow_redirects=False)
                if r.status_code in (301, 302, 403, 429):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status(); j = r.json()
                return j.get("Table", []) if isinstance(j, dict) else (j or [])
            except Exception as e:
                wait = 2 ** (attempt + 1) + random.uniform(0.5, 3)
                print(f"  {e} — retry {attempt + 1}/3 in {wait:.0f}s"); time.sleep(wait); self._seed()
        raise RuntimeError(f"BSE API failed for {params}")

    def announcements(self, code: str, start: date, end: date, category: str | None = None) -> list[dict]:
        rows, cat = [], category if category is not None else self.category
        for page in range(1, 60):
            got = self._get({"pageno": page, "strCat": cat, "strPrevDate": f"{start:%Y%m%d}", "strScrip": code, "strSearch": "P",
                             "strToDate": f"{end:%Y%m%d}", "strType": "C", "subcategory": "-1"})
            rows += got
            if len(got) < 20: break
            self._sleep()
        return rows

    def verify(self, symbol: str) -> str | None:
        """Company name BSE returns for this scrip code — and whether it matches the expected name."""
        rows = self.announcements(BSE_CODE[symbol], date(2025, 1, 1), date(2025, 12, 31), "-1")[:5]
        name = rows[0].get("SLONGNAME", "") if rows else ""
        if not name: print(f"  {symbol}: no filings returned for 2025 — check code {BSE_CODE[symbol]}")
        elif NAME_CHECK[symbol] not in name.lower(): print(f"  {symbol}: code {BSE_CODE[symbol]} is '{name}' — MISMATCH, skipping"); return None
        return name

    def probe_category(self):
        """Does the API accept the category filter? TCS 2023 has 4 transcripts; if the filtered query finds none, query
        all categories instead (≈3× more pages, same result)."""
        rows = self.announcements(BSE_CODE["TCS"], date(2023, 1, 1), date(2023, 12, 31))
        n = sum(is_transcript(r.get("SUBCATNAME", ""), r.get("HEADLINE", "")) for r in rows)
        if n == 0:
            print(f"  category filter '{self.category}' returned {len(rows)} rows / 0 transcripts for TCS 2023 → using all categories")
            self.category = "-1"
        else: print(f"  category filter OK ({n} TCS transcripts in 2023)")

    def _pdf(self, att: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 1000: return True
        for u in PDF_URLS:
            try:
                r = self.s.get(u.format(att), headers={**self.h, "Accept": "application/pdf,*/*"}, timeout=60)
                if r.ok and r.content[:4] == b"%PDF": dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(r.content); return True
            except Exception: pass
            self._sleep(0.5)
        return False

    def run(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        out_pq = self.dir / "bse_filings.parquet"
        have = pd.read_parquet(out_pq) if out_pq.exists() else pd.DataFrame()
        self.probe_category(); new = []
        for sym in symbols:
            if sym not in BSE_CODE: print(f"{sym}: not in universe"); continue
            if not any(k.startswith(f"{sym}|") for k in self.state["done"]) and self.verify(sym) is None: continue
            for y in range(start.year, end.year + 1):
                key = f"{sym}|{y}"
                if key in self.state["done"] and y < date.today().year: continue
                a, b = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
                rows = self.announcements(BSE_CODE[sym], a, b)
                tr = [r for r in rows if is_transcript(r.get("SUBCATNAME", ""), r.get("HEADLINE", ""))]
                ok = 0
                for r in tr:
                    att = (r.get("ATTACHMENTNAME") or "").strip()
                    ts = pd.to_datetime(r.get("DT_TM") or r.get("NEWS_DT"), errors="coerce")
                    if not att or pd.isna(ts): continue
                    rel = f"data/calls/pdfs/{sym}/{ts:%Y%m%d}_{r.get('NEWSID', '')[:8]}.pdf"
                    if self._pdf(att, self.root / rel): ok += 1
                    else: rel = None
                    new.append({"symbol": sym, "isin": r.get("ISIN_CODE", ""), "date": ts.normalize(), "announced_at": ts,
                                "bse_category": r.get("CATEGORYNAME", ""), "bse_subcategory": r.get("SUBCATNAME", ""),
                                "bse_headline": r.get("HEADLINE", ""), "news_id": r.get("NEWSID", ""), "attachment": att, "path": rel})
                    self._sleep(0.5)
                print(f"{sym} {y}: {len(rows)} filings, {len(tr)} transcripts, {ok} PDFs", flush=True)
                self.state["done"] = sorted(set(self.state["done"]) | {key}); self.state_path.write_text(json.dumps(self.state))
                if new:   # checkpoint after every window
                    df = pd.concat([have, pd.DataFrame(new)], ignore_index=True).drop_duplicates(["symbol", "news_id"], keep="last")
                    df.to_parquet(out_pq, index=False)
                self._sleep(2)
        return pd.read_parquet(out_pq) if out_pq.exists() else pd.DataFrame()

    def extract(self, workers: int = 4) -> int:
        """PDF → text into data/calls/transcripts.jsonl (only paths not already cached)."""
        import logging
        from concurrent.futures import ProcessPoolExecutor
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        df = pd.read_parquet(self.dir / "bse_filings.parquet"); jl = self.dir / "transcripts.jsonl"
        done = {json.loads(l)["p"] for l in jl.open()} if jl.exists() else set()
        todo = [p for p in df["path"].dropna().unique() if p not in done and (self.root / p).exists()]
        n = 0
        with ProcessPoolExecutor(workers) as ex, jl.open("a") as fh:
            for p, k, t in ex.map(_extract_one, [str(self.root / p) for p in todo], todo):
                fh.write(json.dumps({"p": p, "pages": k, "t": t}) + "\n"); n += 1
        print(f"extracted {n} transcripts → {jl}"); return n


def _extract_one(full: str, rel: str):
    import pdfplumber
    try:
        with pdfplumber.open(full) as pdf: return rel, len(pdf.pages), "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    except Exception: return rel, 0, ""
