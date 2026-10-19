"""python -m finbert_vol.live <command>

  backfill   --state DIR [--start 2020-01-01]   build the initial state on your Mac: prices from data/prices.csv (+ Yahoo
                                                top-up), Google News headlines from --start, FinBERT scores, outputs.
                                                Then: npx wrangler r2 object put finbert-vol/state.tar.gz --file DIR.tar.gz --remote
  run        --state DIR                        one daily update in place (what the container does)
  pack       --state DIR                        DIR → DIR.tar.gz
  serve      [--port 8080]                      container HTTP server
  check-news [--days 7]                         fetch the last N days of headlines and print them (no state, no model)
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(prog="finbert_vol.live")
    ap.add_argument("cmd", choices=["backfill", "run", "pack", "serve", "check-news"])
    ap.add_argument("--state", default=str(ROOT / "data" / "live_state")); ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--port", type=int, default=8080); ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--target", default="TCS"); ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()

    if a.cmd == "serve":
        from .server import serve; return serve(a.port)
    if a.cmd == "check-news":
        from .gnews import GoogleNewsRSS
        df = GoogleNewsRSS(a.target, delay=a.delay).fetch(date.today() - timedelta(days=a.days), date.today())
        print(df[["date", "source", "title"]].to_string(index=False, max_colwidth=100)); print(f"\n{len(df)} headlines"); return

    from .runner import Runner, State
    st = State(Path(a.state))
    if a.cmd == "pack":
        out = Path(a.state).with_suffix(".tar.gz"); out.write_bytes(st.pack()); print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)"); return

    from finbert_vol.models import SentimentModel
    model = SentimentModel.from_settings(ROOT / "models" / "finbert-in-sentiment", ROOT / "results" / "sentiment" / "calibration.json", "C")
    r = Runner(st, model, target=a.target)
    from .gnews import GoogleNewsRSS
    r.news_src = GoogleNewsRSS(a.target, delay=a.delay)
    if a.cmd == "backfill":
        if not (st.root / "prices.csv").exists() and (ROOT / "data" / "prices.csv").exists():
            p = pd.read_csv(ROOT / "data" / "prices.csv"); p = p[p.Symbol.isin([r.symbol, r.market])]
            p["Date"] = p["Date"].astype(str).str[:10]
            from .runner import PRICE_COLS
            p.reindex(columns=PRICE_COLS).to_csv(st.root / "prices.csv", index=False); print(f"prices: seeded {len(p)} rows from data/prices.csv")
        try: r.update_prices()
        except Exception as e: print(f"prices: Yahoo top-up failed ({e}); using data/prices.csv as is")
        r.update_news(start=date.fromisoformat(a.start))
        r.publish()
        h = pd.read_csv(st.root / "headlines.csv", parse_dates=["date"]); rel = h[h.relevance > 0]
        print("\nrelevant headlines per year:", rel.groupby(rel.date.dt.year).size().to_dict())
        print("by source:", rel.source.value_counts().to_dict())
        out = Path(a.state).with_suffix(".tar.gz"); out.write_bytes(st.pack()); print(f"\nstate → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    elif a.cmd == "run":
        print(r.daily())


if __name__ == "__main__":
    main()
