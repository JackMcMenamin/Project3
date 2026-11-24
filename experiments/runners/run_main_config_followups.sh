#!/usr/bin/env bash
# Follow-ups for the main configuration (pair sampling, 10k THINGS, inflected
# SimVerb, SimLex pairs excluded, split sentences). In order:
#
#   1. MEN on the saved checkpoints: main configuration, and word sampling at
#      the endpoint for comparison
#   2. stem check under the main configuration
#   3. Pereira RSA, seeds 1-3, partitioned by the main configuration's vocabulary
#   4. seeds 2-3 of the pair-sampling-over-all-THINGS and SimVerb-share arms
#   5. GPT-2 under the main configuration, seeds 1-3
#   6. GLUE on main-configuration checkpoints, seeds 1-3
#
# Evaluation steps use checkpoints saved by run_supervision_checks.sh.
# Resumable: a step that finished leaves a marker in $DONE and is skipped.
set -u
cd "$(dirname "$0")/../.." || exit 1

LOG=logs/followups
DONE=$LOG/done
mkdir -p "$DONE"
EVAL=experiments/rsr_pretraining
POOL=artifacts/sentences_strict
MAIN="--exclude-simlex-pairs --split-sentences --sampler pairs --things-pairs 10000 --inflect-verbs"

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

# 1. MEN
step men_main     python $EVAL/eval_men_transfer.py --label main     --runs $(runs inflect pairs 1 2 3 4 5)
step men_endpoint python $EVAL/eval_men_transfer.py --label endpoint --runs $(runs endpoint endpoint 1 2 3 4 5)

# 2. stem check
step stem_main python $EVAL/eval_strict_neither.py --runs $(runs inflect pairs 1 2 3 4 5)

# 3. brain RSA
for s in 1 2 3; do
  step brain_main_s$s bash -c "cd brain_and_glue && python src/eval_pereira_bert.py \
    --models vanilla=bert-base-uncased \
             mlm_control=../models/bert_baseline_sup_pairs_s$s.pt \
             rsr=../models/bert_rsr_sup_inflect_s$s.pt \
    --layers all --skip-2v2 --sup-config ../models/bert_rsr_sup_inflect_s$s.json \
    --output-dir results/brain_main_s$s"
done

# 4. more seeds for the two arms that separate pair sampling from the THINGS cut
step split_arms_s23 env SEEDS=" " SPLIT_SEEDS="2 3" MORE_SEEDS=" " \
  bash experiments/runners/run_supervision_checks.sh

# 5. GPT-2, read out at layer 6 as in the earlier GPT-2 runs
for s in 1 2 3; do
  step gpt2_main_s$s python $EVAL/gpt2_rsr_continued_pretraining.py \
    --regime compare --lambda 0.7 --steps 2000 --eval-every 100 --seed $s \
    --rsr-layer 6 --sentences-dir $POOL --save-tag sup_main_s$s $MAIN
done

# 6. GLUE: vanilla once, then control and RSR per checkpoint
for s in 1 2 3; do
  vanilla=""; [ $s = 1 ] && vanilla="vanilla=bert-base-uncased"
  step glue_main_s$s bash -c "cd brain_and_glue && python src/eval_downstream_bert.py \
    --models $vanilla mlm_control=../models/bert_baseline_sup_pairs_s$s.pt \
             rsr=../models/bert_rsr_sup_inflect_s$s.pt \
    --tasks mrpc rte cola stsb sst2 --seeds 1 2 3 \
    --output-dir results/downstream_main_s$s"
done

echo "all steps attempted at $(date '+%d %b %H:%M')"
