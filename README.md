# FinBERT × TCS volatility — news pipeline

**Question:** does reading TCS news, with a FinBERT model domain-adapted to Indian financial text, improve next-day volatility forecasts beyond what GARCH extracts from price history alone?

## Reproduce with Docker
```bash
git clone https://github.com/seshjuansauce/FinBERT-Indian-Stock-Market-Analysis.git
cd FinBERT-Indian-Stock-Market-Analysis
cp .env.example .env              # add HUGGINGFACE_API_KEY (needed for the news-corpus download)
docker compose build
docker compose run --rm finbert   # prices -> news -> MLM corpus -> features -> volatility ladder
```
| Command | What it does |
|---|---|
| `docker compose run --rm finbert prices` | run one stage (`prices`, `news`, `corpus`, `dapt`, `features`, `ladder`) |
| `docker compose run --rm -e RUN_DAPT=1 finbert` | also retrain the domain-adapted FinBERT (CPU: many hours; ~1.5 h on an Apple-silicon GPU outside Docker) |
| `docker compose up notebook` | JupyterLab at http://localhost:8888 for the notebooks |

`data/` and `models/` are mounted from the host, so downloads and weights persist. Raw news text is never committed (licensing); only code and derived daily features are.


## Setup (once)
```bash
cd ~/Desktop/Projects/FinBERT-Indian-Stock-Market-Analysis
python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml pandas numpy torch transformers
```
`.env` holds `NEWSDATAIO_API_KEY=<newsdata.io key>` and `HUGGINGFACE_API_KEY=<hf token>` and is git-ignored.

## 1 · Pull news (`news/fetch_newsdata.py`)
| Plan | Command | What you get |
|---|---|---|
| Free | `python news/fetch_newsdata.py` | `/latest` = past 48 h only. **Run daily** (cron below) to accumulate history |
| Basic / Professional | `python news/fetch_newsdata.py --archive 2024-10-06 2026-10-06` | `/archive`: 6 months (Basic) / 2 years (Professional) |
| Any | `--dry-run` | prints requests (key redacted), spends nothing |

Companies and keywords: `news/config.yaml` (TCS + Infosys, Wipro, HCLTech, Tech Mahindra + Indian-IT sector).
"TCS" also means *Tax Collected at Source* in Indian news — `exclude` drops those.
Output appends to `data/news_raw.csv` (de-duplicated on `article_id`).

Daily run (free plan) — `run_daily.sh` fetches then scores. Install with `crontab -e` and add:
```
55 17 * * * /Users/sukritgupta/Desktop/Projects/FinBERT-Indian-Stock-Market-Analysis/run_daily.sh
```
(17:55 IST daily. The Mac must be awake; macOS may ask to grant `cron` Full Disk Access the first time —
System Settings → Privacy & Security → Full Disk Access → add `/usr/sbin/cron`.) Log: `data/daily.log`.

## 2 · Entity-aware scoring (`news/entity.py` → `news/score_news.py`)
`python news/score_news.py`. For every (article, company):

| Field | Meaning |
|---|---|
| `title_norm` | source suffixes/prefixes stripped (dedupe key) |
| `role` | primary / co_mention / list_member / peripheral / wrong_entity / sector |
| `relevance` | 1.0 / 0.5 / 0.2 / 0.1 / 0 (non-business and wrong entity = 0) |
| `article_type` | market_wrap, live_updates, stocks_to_watch, buy_sell_reco, results_date, results_preview, results, deal_win, analyst_view, dividend, management, legal, non_business, other |
| `target_span` | the clause about THIS company ("HCLTech drags"); whole title for lists; text after the colon for "Opening Bell Updates:" stubs |
| `fb_t_*` / `fb_f_*` | FinBERT on the target span / on the full title (baseline) |
| `move_signal` | ±1 from price-move words in the target span (drags, worst hit, top gainers, rallies…) — FinBERT misses these |
| `sent_adj` | `move_signal` if non-zero, else `fb_t_net` |
| `reco_signal`, `analyst_signal` | ±1 buy/sell lists, broker views (kept separate from sentiment) |

`data/news_daily.csv` (company × day): **attention** (`n_articles, n_syndicated, n_primary, n_sources, n_results_news, n_market_wrap`),
**entity sentiment** (`sent_target_w, sent_adj_w` relevance-weighted; `sent_*_min` over company-focused items only; `sent_primary`),
**explicit signals** (`move_net, n_move_pos/neg, reco_net, n_buy, n_sell, analyst_net`), and `sent_full_mean` (naive baseline).
Articles at/after 15:30 IST count toward the next trading day.

## 3 · Check against the gold set
`data/gold_headlines.csv` = sentiment toward the target company (labeler `claude_draft` — validate and fill `validated_by_user`).
`score_news.py` prints accuracy / non-neutral accuracy / sign flips for: full headline vs target span vs target + move words.

## Next
Join `news_daily.csv` with TCS daily prices → GARCH baseline σ̂ + news features → realised-volatility model.
