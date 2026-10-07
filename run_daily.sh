#!/bin/zsh
# Daily news pull + entity-aware FinBERT scoring. Scheduled via cron (see README).
cd "$(dirname "$0")"
source .venv/bin/activate
echo "=== $(date '+%Y-%m-%d %H:%M') ===" >> data/daily.log
python news/fetch_newsdata.py >> data/daily.log 2>&1
python news/score_news.py     >> data/daily.log 2>&1
