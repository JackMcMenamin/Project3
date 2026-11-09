#!/usr/bin/env bash
# BERT, one seed per arm, endpoint model (no early stopping).
#
#   endpoint  reported configuration
#   clean     + SimLex pairs excluded, train/eval sentence halves
#   pairs     clean + pair-first sampling over a 10k THINGS sample
#   inflect   pairs + inflected SimVerb copies
#
# The inflect arm needs sentences for the inflected forms. It waits for the
# harvest manifest written by
#   harvest_sentences_strict.py --targets-file artifacts/inflected_targets.txt
#
# Resumable: finished arms are skipped.
set -u
cd "$(dirname "$0")/../.." || exit 1

SEED=${SEED:-1}
STEPS=2000
EVAL_EVERY=100
POOL=artifacts/sentences_strict
SCRIPT=experiments/rsr_pretraining/bert_rsr_continued_pretraining.py
HARVEST_DONE=artifacts/sentences_strict_inflected_targets_manifest.csv
LOG=logs
mkdir -p "$LOG"

CLEAN="--exclude-simlex-pairs --split-sentences"
PAIRS="$CLEAN --sampler pairs --things-pairs 10000"

done_already() {
  ls results/bert_continued_rsr_"$1"_*.csv >/dev/null 2>&1 &&
    ls results/bert_continued_baseline_"$1"_*.csv >/dev/null 2>&1
}

run_arm() {  # run_arm <name> <flags...>
  local name=$1; shift
  local tag="sup_${name}_s${SEED}"
  if done_already "$tag"; then
    echo "=== $name: already complete, skipping ==="
    return
  fi
  echo "=== $name: $(date +%H:%M) ==="
  python "$SCRIPT" --regime compare --lambda 0.7 \
    --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$SEED" \
    --sentences-dir "$POOL" --save-tag "$tag" "$@" \
    > "$LOG/${tag}.log" 2>&1
  echo "  exit $? at $(date +%H:%M)"
}

run_arm endpoint
run_arm clean $CLEAN
run_arm pairs $PAIRS

while [ ! -f "$HARVEST_DONE" ]; do
  echo "waiting for inflected-form harvest ($(date +%H:%M))"
  sleep 600
done
run_arm inflect $PAIRS --inflect-verbs

echo "done. results in results/, logs in $LOG"
