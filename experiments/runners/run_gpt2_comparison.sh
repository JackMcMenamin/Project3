#!/usr/bin/env bash
# The main RSR pipeline, on GPT-2.
#
# Everything else in the project is BERT, which is its largest limitation. This
# asks the central question of an autoregressive model: does RSR improve the
# pairs it was never supervised on, and not the ones it was?
#
# Three conditions, exactly as for BERT:
#   vanilla    untouched GPT-2, no training      (the step-0 row of every run)
#   control    continued causal LM, no RSR term  (the matched baseline)
#   RSR        continued causal LM + RSR term
#
# Two evaluation sets, both split Both / One / Neither by supervision exposure:
#   SimLex-999   held out, never trained on
#   MEN          never used in training or model selection, and rates
#                relatedness rather than similarity, so it is a real transfer
#                test rather than a second look at the same thing
#
# Held identical to the BERT runs by importing that script rather than
# reimplementing: supervision (SimVerb + THINGS, 2,652 words, MEN excluded),
# the partition, the sentence pool, and the soft-Spearman loss itself. Only the
# model and its anchor change, causal LM in place of masked LM, keeping
#     loss = (1 - lambda) * anchor + lambda * RSR
#
# ONE CHOICE HAD TO DIFFER: the read-out layer. The probe over all 999 pairs
# (results/gpt2_layer_probe_20260908_005631.csv) found GPT-2's final layer
# degenerate - all-pairs 0.161 with cosines at 0.994 +/- 0.006, a space collapsed
# to a needle - while layer 6 reads 0.529. BERT's layer 12 reads 0.480, so it is
# a serviceable lexical read-out and GPT-2's is not. Layer 6 is therefore the
# functional counterpart of BERT's layer 12, and running at 12 would measure
# anisotropy rather than RSR. Layer 12 is one flag away if it is ever wanted.
#
# RESUMABLE: each seed checks for its own output and skips if present.
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS=2000
EVAL_EVERY=200       # 10 SimLex points; MEN is scored at the endpoints.
                     # The 300-step pilot peaked on Neither somewhere in
                     # 200-400, so a coarser grid would risk early stopping
                     # picking a checkpoint either side of the interesting part.
LAYER=6
SEEDS="1 2 3"
POOL=artifacts/sentences_strict
SCRIPT=experiments/rsr_pretraining/gpt2_rsr_continued_pretraining.py
LOG=logs
mkdir -p "$LOG"

for seed in $SEEDS; do
  tag="main_s${seed}"
  if ls results/gpt2_continued_rsr_"$tag"_*.csv >/dev/null 2>&1 \
     && ls results/gpt2_continued_baseline_"$tag"_*.csv >/dev/null 2>&1; then
    echo "=== seed $seed : already complete, skipping ==="
    continue
  fi
  echo "=== seed $seed : vanilla + CLM control + RSR, SimLex and MEN ==="
  python "$SCRIPT" --regime compare --lambda 0.7 \
    --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$seed" \
    --rsr-layer "$LAYER" --sentences-dir "$POOL" --save-tag "$tag" \
    > "$LOG/gpt2_${tag}.log" 2>&1
  echo "  exit $?"
done

echo "done. results in results/gpt2_continued_*, logs in $LOG"
