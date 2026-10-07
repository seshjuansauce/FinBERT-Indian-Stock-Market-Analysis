"""Pull TCS + peer + Indian-IT news from newsdata.io into data/news_raw.csv (append, de-duplicated).

  python news/fetch_newsdata.py                       # FREE plan: /latest (past 48h). Run daily (cron) to build history.
  python news/fetch_newsdata.py --archive 2024-10-01 2026-10-06   # PAID (Basic: 6 mo, Professional: 2 yr): /archive
  python news/fetch_newsdata.py --dry-run             # print the requests (key redacted), spend nothing

Reads NEWSDATAIO_API_KEY from .env (never printed). Free-plan limits respected: 10 articles/credit, 30 credits/15 min, 200/day.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
RAW = DATA / "news_raw.csv"
STATE = DATA / "fetch_state.json"
BASE = "https://newsdata.io/api/1"
FIELDS = ["article_id", "company", "query", "endpoint", "pubDate", "title", "description", "content", "link",
          "source_id", "source_name", "source_priority", "country", "category", "keywords", "language", "fetched_at"]

def load_env() -> dict:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def load_cfg() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "news" / "config.yaml").read_text())

class Budget:
    """Client-side credit pacing: free plan = 30 credits / 15 min, 200 / day (paid: 1800 / 15 min)."""
    def __init__(self, per_15m: int, per_day: int):
        self.per_15m, self.per_day = per_15m, per_day
        st = json.loads(STATE.read_text()) if STATE.exists() else {}
        today = date.today().isoformat()
        self.used_today = st.get("used", 0) if st.get("day") == today else 0
        self.window: list[float] = []
    def take(self, dry: bool = False):
        if dry: return
        if self.used_today >= self.per_day:
            raise SystemExit(f"daily credit budget ({self.per_day}) reached — resume tomorrow (progress is saved)")
        now = time.time(); self.window = [t for t in self.window if now - t < 900]
        if len(self.window) >= self.per_15m:
            wait = 900 - (now - self.window[0]) + 2
            print(f"  rate window full — sleeping {wait:.0f}s", flush=True); time.sleep(wait)
        self.window.append(time.time()); self.used_today += 1
        STATE.write_text(json.dumps({"day": date.today().isoformat(), "used": self.used_today}))

def get(endpoint: str, params: dict, key: str, dry: bool) -> dict:
    url = f"{BASE}/{endpoint}?{urlencode({**params, 'apikey': key})}"
    if dry:
        print("  GET", url.replace(key, "***")); return {"status": "dry", "results": [], "nextPage": None}
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "finbert-research/0.1"}), timeout=30) as r:
                return json.loads(r.read().decode())
        except HTTPError as e:
            body = e.read().decode(errors="ignore")[:300]
            if e.code == 429: print("  429 rate limited — sleeping 60s", flush=True); time.sleep(60); continue
            raise SystemExit(f"HTTP {e.code}: {body}")
        except URLError as e:
            print(f"  network error {e} — retry", flush=True); time.sleep(5 * (attempt + 1))
    raise SystemExit("giving up after retries")

def existing_ids() -> set[str]:
    if not RAW.exists(): return set()
    with RAW.open() as f: return {r["article_id"] for r in csv.DictReader(f)}

def keep(article: dict, comp: dict) -> bool:
    text = f"{article.get('title') or ''} {article.get('description') or ''}"
    if comp.get("must_match") and not re.search(comp["must_match"], text): return False
    if comp.get("exclude") and re.search(comp["exclude"], text): return False
    return True

def run(args):
    cfg = load_cfg(); env = load_env(); key = env.get("NEWSDATAIO_API_KEY", "")
    if not key and not args.dry_run: raise SystemExit("NEWSDATAIO_API_KEY missing in .env")
    paid = bool(args.archive)
    budget = Budget(1800 if paid else 30, 10_000 if paid else 200)
    seen = existing_ids(); new_rows = 0
    endpoint = "archive" if paid else "latest"
    windows = [(None, None)]
    if paid:  # archive in monthly windows so pagination stays shallow
        d0, d1 = date.fromisoformat(args.archive[0]), date.fromisoformat(args.archive[1])
        windows, d = [], d0
        while d <= d1:
            e = min(d + timedelta(days=30), d1); windows.append((d.isoformat(), e.isoformat())); d = e + timedelta(days=1)
    out = Path(os.devnull) if args.dry_run else RAW
    write_header = args.dry_run or not RAW.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header: w.writeheader()
        for name, comp in cfg["companies"].items():
            if args.only and name not in args.only: continue
            q = " OR ".join(comp["title_terms"])
            for (fd, td) in windows:
                params = {"qInTitle": q, "language": cfg.get("language", "en"), "removeduplicate": 1}
                if cfg.get("countries"): params["country"] = ",".join(cfg["countries"])
                if fd: params.update(from_date=fd, to_date=td)
                page, pages = None, 0
                while True:
                    if page: params["page"] = page
                    budget.take(args.dry_run)
                    js = get(endpoint, params, key, args.dry_run)
                    if js.get("status") == "error":
                        print(f"  API error for {name}: {js.get('results')}", flush=True); break
                    res = js.get("results") or []; kept = 0
                    for a in res:
                        if a.get("article_id") in seen or not keep(a, comp): continue
                        a = {**a, "company": name, "query": q, "endpoint": endpoint, "fetched_at": datetime.now().isoformat(timespec="seconds"),
                             "keywords": ";".join(a.get("keywords") or []), "country": ";".join(a.get("country") or []),
                             "category": ";".join(a.get("category") or [])}
                        w.writerow(a); seen.add(a["article_id"]); kept += 1; new_rows += 1
                    f.flush(); pages += 1
                    print(f"{name:8s} {fd or 'latest'}..{td or ''} page {pages}: {len(res)} returned, {kept} kept", flush=True)
                    page = js.get("nextPage")
                    if not page or args.dry_run or pages >= args.max_pages: break
    print(f"\nnew articles: {new_rows} → {RAW}  | credits used today: {budget.used_today}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", nargs=2, metavar=("FROM", "TO"), help="paid plans only: YYYY-MM-DD YYYY-MM-DD")
    ap.add_argument("--only", nargs="*", help="subset of companies, e.g. TCS INFY")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    run(ap.parse_args())
