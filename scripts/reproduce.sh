#!/usr/bin/env bash
# Pipeline runner. Usage: reproduce.sh [all|prices|news|corpus|dapt|features|ladder|STAGE ...]
#   all   = prices news corpus features ladder   (dapt is opt-in: RUN_DAPT=1, slow on CPU)
#   DAPT_MODEL=<hf repo id>  -> use published domain-adapted weights instead of training them
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <stage> <script> [args...]  — skips stages whose script isn't in the repo yet
  local stage=$1 script=$2; shift 2
  if [[ ! -f $script ]]; then echo "── [$stage] skipped: $script not built yet"; return 0; fi
  echo "── [$stage] python $script $*"; python "$script" "$@"
}

stage() {
  case $1 in
    prices)   run prices   prices/update_prices.py ;;                 # yfinance (Postgres cross-check if reachable)
    news)     if [[ -f data/hf/raw_dataset/economictimes_raw.csv ]]; then echo "── [news] data/hf present, skipping download"
              else run news news/download_hf_bucket.py; fi ;;          # needs HUGGINGFACE_API_KEY in .env
    corpus)   run corpus   bert/prepare_mlm_corpus.py ;;
    dapt)     if [[ -n ${DAPT_MODEL:-} ]]; then
                python -c "from huggingface_hub import snapshot_download as s; s('$DAPT_MODEL', local_dir='models/finbert-in-dapt')"
              else run dapt bert/dapt_mlm.py; fi ;;
    features) run features model/build_features.py ;;
    ladder)   run ladder   model/run_ladder.py ;;
    *) echo "unknown stage: $1"; exit 2 ;;
  esac
}

[[ $# -eq 0 ]] && set -- all
for s in "$@"; do
  if [[ $s == all ]]; then
    for t in prices news corpus; do stage $t; done
    [[ ${RUN_DAPT:-0} == 1 || -n ${DAPT_MODEL:-} ]] && stage dapt
    for t in features ladder; do stage $t; done
  else
    stage "$s"
  fi
done
echo "── done"
