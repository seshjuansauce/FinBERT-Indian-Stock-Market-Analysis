# TCS volatility tracker on Cloudflare

```
Mon–Fri 23:10 IST  cron ─► Worker ─► Runner (Durable Object) ─► Container (Dockerfile.live, ½ vCPU / 4 GiB)
                                          │  R2 state/state.tar.gz ──► POST /run ──► prices (Yahoo) + headlines (Google News RSS)
                                          │                                         FinBERT-IN → GJR-GARCH + OLS → forecast
                                          ◄── JSON: public files + new state ──────┘
Browser ─► Worker ─► /           dashboard (public/index.html)
                     /api/*.csv|json   from R2 public/
```

Model: `ln RV_{t+1} = ln h_{t+1|t} (GJR-GARCH) + u`, with `u` regressed walk-forward on HAR terms (own + NIFTY 500 RV) and
FinBERT news features (headline count, no-news flag, mean calibrated sentiment, mean |sentiment|). See `finbert_vol/live/model.py`.

## One-time setup (on your Mac, from the repo root)

**1 · Check the news source works from your network**
```bash
git pull && pip install -r requirements-live.txt
python -m finbert_vol.live check-news --days 7          # prints last week's TCS headlines from ET, Moneycontrol, BS, FE, Mint
```

**2 · Build the starting state** (prices from `data/prices.csv` + Yahoo top-up, Google News headlines since 2020, FinBERT scores, outputs).
About 350 weekly requests at 1–2.5 s each, so ~15–25 min; it can be re-run and only fills what's missing after the first time.
```bash
python -m finbert_vol.live backfill --start 2020-01-01    # → data/live_state/ and data/live_state.tar.gz
```
It prints relevant headlines per year — check the early years aren't near-empty (Google News' archive thins out with age;
if 2020–21 are sparse, use `--start 2022-01-01`: the model still has 2½ years before the test period).

**3 · Cloudflare** (Docker Desktop must be running — `wrangler deploy` builds the container image locally)
```bash
cd cloudflare
npm install
npx wrangler login
npx wrangler r2 bucket create finbert-vol
npx wrangler r2 object put finbert-vol/state/state.tar.gz --file ../data/live_state.tar.gz --remote
for f in tracker.csv latest.json summary.json headlines.json; do
  npx wrangler r2 object put finbert-vol/public/$f --file ../data/live_state/public/$f --remote
done
npx wrangler secret put RUN_TOKEN                          # any long random string; protects the manual-run endpoint
npx wrangler deploy                                        # first build + push of the ~1.3 GB image takes a few minutes
```
The dashboard is live at `https://finbert-vol.<your-subdomain>.workers.dev` straight away (from the uploaded files).
After the first deploy, wait a few minutes for the container image to roll out, then test one real run:
```bash
curl -X POST https://finbert-vol.<your-subdomain>.workers.dev/api/run -H "Authorization: Bearer <RUN_TOKEN>"
curl https://finbert-vol.<your-subdomain>.workers.dev/api/status      # log of the last run
```

## Every day
Nothing to do. The cron runs Mon–Fri at 17:40 UTC (23:10 IST). A failed run (e.g. Yahoo or Google refusing a request)
keeps yesterday's dashboard and state, and `/api/status` shows the error. The previous state is kept as
`state/state.prev.tar.gz`; daily logs are in `logs/YYYY-MM-DD.json`.

## Cost (Workers Paid, $5/month)
One run ≈ 1–3 min on `standard-1`. A month ≈ 22 runs ≈ 60 container-minutes: well inside the included
375 vCPU-min, 25 GiB-h memory and 200 GB-h disk. R2 storage is a few MB (free tier 10 GB). Expected total: **$5/month**.

## Changing things
| What | Where |
|---|---|
| Schedule | `triggers.crons` in `wrangler.jsonc` (UTC) |
| News sites / query | `SITES`, `QUERY_NAMES` in `finbert_vol/live/gnews.py` |
| Model features | `HAR`, `NEWS` in `finbert_vol/live/model.py` |
| Dashboard | `public/index.html` (`npx wrangler deploy` re-uploads it; no image rebuild needed unless Python changed) |
| Local run of the daily job | `python -m finbert_vol.live run --state data/live_state` |
