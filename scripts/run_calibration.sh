#!/bin/zsh
# GPU step for calibration (Mac): fit temperatures on the dev split, then re-score the corpus storing raw logits.
# After this, everything else (calibrated features, ladder) runs on CPU from the stored logits.
set -e; setopt pipefail
cd "$(dirname "$0")/.."
[[ -f ../FinBERT/.venv/bin/activate ]] && source ../FinBERT/.venv/bin/activate
LOG=results/sentiment/calibration_log.txt; : > $LOG
log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a $LOG; }
log "temperature scaling (A, B, C)"; python -u bert/calibrate.py 2>&1 | tee -a $LOG
log "re-score corpus with FinBERT-IN (C), storing logits"; python news/score_corpus.py --model models/finbert-in-sentiment --tag C 2>&1 | tee -a $LOG
log "re-score corpus with off-the-shelf FinBERT (A), storing logits"; python news/score_corpus.py --model ProsusAI/finbert --tag A 2>&1 | tee -a $LOG
log "DONE"
