"""python -m finbert_vol <step> [--config finbert_vol.yaml] [--target TCS]

steps
  score-news    FinBERT-IN (calibrated) on every TCS article → data/features/articles_tcs.csv           (GPU helps)
  score-calls   call model on TCS/INFY/WIPRO transcripts → data/calls/{call_sentences_scored,calls}.csv  (GPU helps)
  features      daily model table (prices + news + announcement flags + live call features)           (CPU)
  evaluate      walk-forward rungs, QLIKE + Diebold–Mariano → results/pipeline/evaluation.csv           (CPU)
  call-test     pooled call-window test (u over 1/5/10 days after publication) → results/pipeline/call_test.csv
  predict       next-day forecast from the latest row → results/pipeline/prediction_latest.json
  all           score-news, score-calls, features, evaluate, call-test, predict
  fetch-transcripts  NIFTY 50 transcripts from BSE → data/calls/{bse_filings.parquet,pdfs/,transcripts.jsonl}  (run on your Mac)

  --targets ALL       score-calls / call-test over every company with transcripts and price features
"""
import argparse

from . import Pipeline, Settings
from .features import CALL_FEATURES
from .volatility import Evaluator


def main():
    ap = argparse.ArgumentParser(prog="finbert_vol")
    ap.add_argument("step", choices=["score-news", "score-calls", "features", "evaluate", "call-test", "predict", "all", "fetch-transcripts"])
    ap.add_argument("--config", default=None); ap.add_argument("--target", default=None)
    ap.add_argument("--targets", default=None, help="comma list or ALL (call steps)")
    ap.add_argument("--symbols", default=None, help="fetch-transcripts: comma list (default: NIFTY 50)")
    ap.add_argument("--start", default="2021-01-01"); ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    s = Settings.load(a.config)
    if a.target: s.target = a.target
    if a.targets: s.call_targets = a.targets.split(",")
    if a.step == "fetch-transcripts":
        from datetime import date
        from .bse_fetch import BSETranscriptFetcher
        from .universe import NIFTY50
        f = BSETranscriptFetcher(s.path("."), delay=a.delay)
        syms = a.symbols.split(",") if a.symbols else list(NIFTY50)
        df = f.run(syms, date.fromisoformat(a.start), date.today()); f.extract()
        print(df.groupby("symbol").size().sort_values().to_string() if len(df) else "nothing fetched"); return
    p = Pipeline(s)
    steps = ["score-news", "score-calls", "features", "evaluate", "call-test", "predict"] if a.step == "all" else [a.step]
    for st in steps:
        print(f"== {st}")
        if st == "score-news": p.score_news()
        elif st == "score-calls": p.score_calls()
        elif st == "features":
            for t in p.call_universe():
                try: p.build_features(t); print(f"   model table for {t}")
                except FileNotFoundError as e: print(f"   {t}: {e}")
        elif st == "evaluate": print(p.evaluate().round(4).to_string(index=False))
        elif st == "call-test":
            panel = p.call_window_panel(); panel.to_csv(p.out / "call_panel.csv", index=False)
            feats = [f for f in CALL_FEATURES if f in panel]
            res = [Evaluator.call_window_test(panel, feats, f"u_{h}d").assign(horizon=f"{h}d") for h in (1, 5, 10)]
            import pandas as pd
            out = pd.concat(res, ignore_index=True); out.to_csv(p.out / "call_test.csv", index=False)
            print(out.round(3).to_string(index=False))
        elif st == "predict": print(p.predict_latest())


if __name__ == "__main__":
    main()
