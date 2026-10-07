"""Historical news headlines from the GDELT DOC 2.0 API (free, no key) -> data/news_raw.csv (same schema as newsdata).

  python news/fetch_gdelt.py                          # 2021-01-01 -> today, all companies in news/config.yaml
  python news/fetch_gdelt.py --from 2024-07-01 --to 2024-07-31 --only TCS
  python news/fetch_gdelt.py --dry-run

Weekly windows; a window that hits the 250-record cap is split in half recursively. ~6 s between calls (GDELT rate-limits).
Resumable: completed (company, window) pairs are recorded in data/gdelt_state.json.
English + Indian sources by default. GDELT gives title/url/date/domain only (no body) — entity scoring works on titles.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
RAW = DATA / "news_raw.csv"; STATE = DATA / "gdelt_state.json"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
FIELDS = ["article_id", "company", "query", "endpoint", "pubDate", "title", "description", "content", "link",
          "source_id", "source_name", "source_priority", "country", "category", "keywords", "language", "fetched_at"]
QUERIES = {   # GDELT syntax: quoted phrases, OR inside parentheses; short tokens like "TCS" need care (min length 3 ok)
    "TCS": '("Tata Consultancy Services" OR TCS)',
    "INFY": "Infosys",
    "WIPRO": "Wipro",
    "HCLTECH": '("HCL Technologies" OR HCLTech OR "HCL Tech")',
    "TECHM": '"Tech Mahindra"',
    "SECTOR": '("Nifty IT" OR "IT stocks" OR "Indian IT")',
}
SLEEP = 6.0

def call(params: dict, dry: bool) -> dict:
    url = f"{API}?{urlencode(params)}"
    if dry:
        print("  GET", url); return {"articles": []}
    for attempt in range(6):
        try:
            with urlopen(Request(url, headers={"User-Agent": "finbert-research/0.1"}), timeout=60) as r:
                body = r.read().decode("utf-8", errors="ignore").strip()
            time.sleep(SLEEP)
            if not body: return {"articles": []}
            if not body.startswith("{"):   # GDELT returns plain-text errors (e.g. query too short / timespan)
                print(f"  GDELT message: {body[:160]}"); return {"articles": []}
            return json.loads(body)
        except HTTPError as e:
            wait = 30 * (attempt + 1) if e.code == 429 else 10
            print(f"  HTTP {e.code} — sleeping {wait}s", flush=True); time.sleep(wait)
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"  {type(e).__name__} — retry", flush=True); time.sleep(10 * (attempt + 1))
    raise SystemExit("GDELT unreachable after retries — re-run later (progress is saved)")

def fetch_window(company: str, q: str, d0: datetime, d1: datetime, dry: bool, depth: int = 0) -> list[dict]:
    params = {"query": f"{q} sourcelang:english sourcecountry:india", "mode": "artlist", "format": "json",
              "maxrecords": 250, "sort": "datedesc",
              "startdatetime": d0.strftime("%Y%m%d%H%M%S"), "enddatetime": d1.strftime("%Y%m%d%H%M%S")}
    arts = call(params, dry).get("articles", []) or []
    if len(arts) >= 250 and (d1 - d0) > timedelta(hours=6) and depth < 6:   # capped -> split
        mid = d0 + (d1 - d0) / 2
        return fetch_window(company, q, d0, mid, dry, depth + 1) + fetch_window(company, q, mid, d1, dry, depth + 1)
    return arts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default="2021-01-01"); ap.add_argument("--to", dest="d1", default=date.today().isoformat())
    ap.add_argument("--only", nargs="*"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    seen = set()
    if RAW.exists():
        with RAW.open() as f: seen = {r["article_id"] for r in csv.DictReader(f)}
    out = open("/dev/null", "a") if a.dry_run else RAW.open("a", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    if not a.dry_run and RAW.stat().st_size == 0: w.writeheader()
    start, end = datetime.fromisoformat(a.d0), datetime.fromisoformat(a.d1)
    total = 0
    for comp, q in QUERIES.items():
        if a.only and comp not in a.only: continue
        d = start
        while d < end:
            e = min(d + timedelta(days=7), end); key = f"{comp}|{d:%Y%m%d}|{e:%Y%m%d}"
            if key in state and not a.dry_run: d = e; continue
            arts = fetch_window(comp, q, d, e, a.dry_run); kept = 0
            for x in arts:
                aid = "gd_" + hashlib.sha1((x.get("url", "") + comp).encode()).hexdigest()[:16]
                if aid in seen: continue
                seen.add(aid); kept += 1
                sd = x.get("seendate", "")      # e.g. 20240711T141500Z (UTC)
                pub = datetime.strptime(sd, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d %H:%M:%S") if sd else ""
                w.writerow({"article_id": aid, "company": comp, "query": q, "endpoint": "gdelt", "pubDate": pub,
                            "title": x.get("title", ""), "description": "", "content": "", "link": x.get("url", ""),
                            "source_id": x.get("domain", ""), "source_name": x.get("domain", ""), "country": x.get("sourcecountry", ""),
                            "language": x.get("language", ""), "fetched_at": datetime.now().isoformat(timespec="seconds")})
            out.flush(); total += kept
            print(f"{comp:8s} {d:%Y-%m-%d}→{e:%Y-%m-%d}: {len(arts):3d} returned, {kept:3d} new", flush=True)
            if not a.dry_run:
                state[key] = len(arts); STATE.write_text(json.dumps(state))
            if a.dry_run: break
            d = e
    print(f"\nnew articles: {total}")

if __name__ == "__main__":
    main()
