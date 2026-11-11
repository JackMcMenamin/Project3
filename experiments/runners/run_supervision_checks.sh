#!/usr/bin/env bash
# BERT, endpoint model (no early stopping).
#
#   endpoint    reported configuration
#   clean       + SimLex pairs excluded, train/eval sentence halves
#   pairs       clean + pair-first sampling over a 10k THINGS sample
#   inflect     pairs + inflected SimVerb copies
#   pairs_full  clean + pair-first sampling over all THINGS pairs
#   share33     pairs_full with a third of drawn pairs from SimVerb
#
# The control does not use the RSR sampler, so arms marked rsr-only are
# compared against the control of the arm with the same SimLex partition:
# inflect against pairs (10k THINGS), pairs_full and share33 against clean.
#
# The inflect arm needs sentences for the inflected forms. It waits for the
# harvest manifest written by
#   harvest_sentences_strict.py --targets-file artifacts/inflected_targets.txt
#
# Resumable: finished arms are skipped.
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS=2000
EVAL_EVERY=100
POOL=artifacts/sentences_strict
SCRIPT=experiments/rsr_pretraining/bert_rsr_continued_pretraining.py
HARVEST_DONE=artifacts/sentences_strict_inflected_targets_manifest.csv
LOG=logs
mkdir -p "$LOG"

CLEAN="--exclude-simlex-pairs --split-sentences"
PAIRS="$CLEAN --sampler pairs --things-pairs 10000"
PAIRS_FULL="$CLEAN --sampler pairs"

have() { ls results/bert_continued_"$1"_"$2"_*.csv >/dev/null 2>&1; }

run_arm() {  # run_arm <seed> <compare|rsr> <name> <flags...>
  local seed=$1 regime=$2 name=$3; shift 3
  local tag="sup_${name}_s${seed}"
  if have rsr "$tag" && { [ "$regime" = rsr ] || have baseline "$tag"; }; then
    echo "=== $name s$seed: already complete, skipping ==="
    return
  fi
  if [ "$name" = inflect ]; then
    while [ ! -f "$HARVEST_DONE" ]; do
      echo "waiting for inflected-form harvest ($(date +%H:%M))"
      sleep 600
    done
  fi
  echo "=== $name s$seed ($regime): $(date +%H:%M) ==="
  python "$SCRIPT" --regime "$regime" --lambda 0.7 \
    --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$seed" \
    --sentences-dir "$POOL" --save-tag "$tag" "$@" \
    > "$LOG/${tag}.log" 2>&1
  echo "  exit $? at $(date +%H:%M)"
}

main_arms() {  # main_arms <seed>
  run_arm "$1" compare endpoint
  run_arm "$1" compare clean $CLEAN
  run_arm "$1" compare pairs $PAIRS
  run_arm "$1" rsr inflect $PAIRS --inflect-verbs
}

for seed in ${SEEDS:-1 2 3}; do main_arms "$seed"; done
for seed in ${SPLIT_SEEDS:-1}; do
  run_arm "$seed" rsr pairs_full $PAIRS_FULL
  run_arm "$seed" rsr share33 $PAIRS_FULL --simverb-share 0.33
done
for seed in ${MORE_SEEDS:-4 5}; do main_arms "$seed"; done

echo "done. results in results/, logs in $LOG"
