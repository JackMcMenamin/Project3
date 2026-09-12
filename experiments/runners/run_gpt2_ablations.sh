#!/usr/bin/env bash
# Two GPT-2 ablations, both answering questions the main run leaves open.
#
# 1. THE RSR WEIGHT WAS INHERITED, NOT CHOSEN.
#    lambda = 0.7 came from the BERT sweep over 0.3/0.5/0.7/0.9. Holding it fixed
#    made the architectures comparable, which was the right call for the headline,
#    but it means nobody has checked whether 0.7 suits a causal model. The GPT-2
#    gain is a third of BERT's on SimLex, and "you inherited the hyperparameter"
#    is the first thing anyone will say about that. This runs the same sweep.
#
#    The control does not use lambda at all, so seed 1's existing baseline is the
#    correct comparison for every arm here and is not re-run.
#
# 2. POOLING IS NOT OBVIOUSLY RIGHT UNDER CAUSAL ATTENTION.
#    We mean-pool a word's sub-word tokens, matching BERT exactly. But in a causal
#    model only the LAST sub-word token has seen the whole word; the earlier ones
#    saw a prefix. Mean-pooling therefore averages complete and incomplete views.
#    Words here are 1.56 sub-word pieces on average and up to 4, so this is not a
#    corner case. Needs its own control, since the pooling also changes evaluation.
#
# Both are single-seed: they are ablations meant to show whether the main
# configuration was reasonable, not results to quote. Anything that looks
# decisive gets re-run with three seeds before it goes anywhere.
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS=2000
EVAL_EVERY=200
LAYER=6
SEED=1
POOL=artifacts/sentences_strict
SCRIPT=experiments/contextual_finetune/gpt2_rsr_continued_pretraining.py
LOG=RSR_July/logs
mkdir -p "$LOG"

# ---- 1. lambda sweep (0.7 already exists as the main run) -----------------
for lam in 0.3 0.5 0.9; do
  tag="lam${lam/./}_s${SEED}"
  if ls results/gpt2_continued_rsr_"$tag"_*.csv >/dev/null 2>&1; then
    echo "=== lambda $lam : already complete, skipping ==="
    continue
  fi
  echo "=== lambda $lam : RSR only, control reused from the main run ==="
  python "$SCRIPT" --regime rsr --lambda "$lam" \
    --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$SEED" \
    --rsr-layer "$LAYER" --sentences-dir "$POOL" --save-tag "$tag" \
    > "$LOG/gpt2_${tag}.log" 2>&1
  echo "  exit $?"
done

# ---- 2. last-token pooling, with its own matched control -----------------
tag="poollast_s${SEED}"
if ls results/gpt2_continued_rsr_"$tag"_*.csv >/dev/null 2>&1 \
   && ls results/gpt2_continued_baseline_"$tag"_*.csv >/dev/null 2>&1; then
  echo "=== last-token pooling : already complete, skipping ==="
else
  echo "=== last-token pooling : control + RSR ==="
  python "$SCRIPT" --regime compare --lambda 0.7 --pool last \
    --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$SEED" \
    --rsr-layer "$LAYER" --sentences-dir "$POOL" --save-tag "$tag" \
    > "$LOG/gpt2_${tag}.log" 2>&1
  echo "  exit $?"
fi

echo "done. results in results/gpt2_continued_*, logs in $LOG"
