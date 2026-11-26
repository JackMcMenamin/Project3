#!/usr/bin/env bash
# SimVerb share with all of THINGS kept, and what it costs outside SimLex.
# In order:
#
#   1. MEN on pairs_full and share33, seeds 1-3, against the clean control
#   2. share 0.10 and 0.66 (pairs over all THINGS), seeds 1-3, RSR only
#   3. MEN on the new share arms
#   4. GLUE on share33 against the clean control, seeds 1-3
#   5. MEN on the pairs arm, seeds 1-5
#
# With pairs_full (share 0, about 0.03 SimVerb pairs per batch) and share33
# (4.6 per batch) this gives four points; 0.66 gives about 10, above the
# main configuration's 7.4 SimVerb pairs per batch.
# Resumable: a step that finished leaves a marker in $DONE and is skipped.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOG=logs/simverb_share
DONE=$LOG/done
mkdir -p "$DONE"
EVAL=experiments/rsr_pretraining
SCRIPT=$EVAL/bert_rsr_continued_pretraining.py
POOL=artifacts/sentences_strict
export USE_TF=0
PAIRS_FULL="--exclude-simlex-pairs --split-sentences --sampler pairs"

step() {  # step <name> <command...>
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then echo "=== $name: done, skipping ==="; return; fi
  echo "=== $name: $(date '+%d %b %H:%M') ==="
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$DONE/$name"; echo "  ok at $(date +%H:%M)"
  else
    echo "  FAILED (exit $?) at $(date +%H:%M), see $LOG/$name.log"
  fi
}

runs() {  # runs <rsr_arm> <control_arm> <seeds...>
  local r=$1 c=$2; shift 2
  for s in "$@"; do printf "sup_%s_s%s:sup_%s_s%s " "$r" "$s" "$c" "$s"; done
}

# 1. MEN on the existing arms
step men_split python $EVAL/eval_men_transfer.py --label split \
  --runs $(runs pairs_full clean 1 2 3) $(runs share33 clean 1 2 3)

# 2. share sweep
for s in 1 2 3; do
  for share in 10 66; do
    step share${share}_s$s python $SCRIPT --regime rsr --lambda 0.7 \
      --steps 2000 --eval-every 100 --seed $s --sentences-dir $POOL \
      --save-tag sup_share${share}_s$s $PAIRS_FULL --simverb-share 0.$share
  done
done

# 3. MEN on the sweep
step men_sweep python $EVAL/eval_men_transfer.py --label sweep \
  --runs $(runs share10 clean 1 2 3) $(runs share66 clean 1 2 3)

# 4. GLUE on share33
for s in 1 2 3; do
  step glue_share33_s$s bash -c "cd brain_and_glue && python src/eval_downstream_bert.py \
    --models mlm_control=../models/bert_baseline_sup_clean_s$s.pt \
             rsr=../models/bert_rsr_sup_share33_s$s.pt \
    --tasks mrpc rte cola stsb sst2 --seeds 1 2 3 \
    --output-dir results/downstream_share33_s$s"
done

# 5. MEN on the pairs arm (10k THINGS, no inflection), to separate the THINGS
#    cut from the inflected forms as the source of the MEN cost
step men_pairs python $EVAL/eval_men_transfer.py --label pairs --runs $(runs pairs pairs 1 2 3 4 5)
echo "all steps attempted at $(date '+%d %b %H:%M')"
