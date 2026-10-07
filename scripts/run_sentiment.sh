#!/bin/zsh
# GPU part of the pipeline (run on the Mac): labels -> fine-tune A/B/C -> score the corpus with A and C.
# Everything is logged to results/sentiment/run_log.txt so progress can be followed from another session.
set -e; setopt pipefail
cd "$(dirname "$0")/.."
[[ -f ../FinBERT/.venv/bin/activate ]] && source ../FinBERT/.venv/bin/activate
mkdir -p results/sentiment; LOG=results/sentiment/run_log.txt; : > $LOG
log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a $LOG; }
log "installing light deps"; pip install -q pyyaml duckdb huggingface_hub pyarrow 2>&1 | tail -1 | tee -a $LOG
log "labels";   python bert/build_label_sets.py 2>&1 | tee -a $LOG
log "fine-tune ${SENTIMENT_ARGS:---quick}"; caffeinate -i python -u bert/finetune_sentiment.py ${=SENTIMENT_ARGS:---quick} 2>&1 | tee -a $LOG
log "score corpus with FinBERT-IN (C)"; python news/score_corpus.py --model models/finbert-in-sentiment --tag C 2>&1 | tee -a $LOG
log "score corpus with off-the-shelf FinBERT (A)"; python news/score_corpus.py --model ProsusAI/finbert --tag A 2>&1 | tee -a $LOG
log "DONE"
