#!/usr/bin/env bash
# Template versus real-sentence RSR, with evaluation context decoupled.
#
# The two original RSR arms differ by one flag, `use_templates` (samplers.py:107 there),
# and they behave completely differently: rsr_all rises on every partition while
# rsr_targetwords collapses early and only recovers over thousands of steps.
#
# The original design confounds two things, because evaluation context is a SECOND flag
# (eval_simlex.py) that defaults to scoring words in isolation:
#
#   template training  + bare evaluation  -> trained and scored on near-identical input
#   sentence training  + bare evaluation  -> trained on context, scored without it
#
# So the collapse could be a training effect or a measurement artefact, and the original
# figure cannot tell them apart. This runs the training arms in our pipeline and
# scores every checkpoint in all three contexts, which separates them:
#
#   if sentence-trained RSR looks fine under sentence eval and bad under bare
#   eval, the collapse is the measurement. If it is bad under both, it is real.
#
# Arms per seed:
#   A  compare, real sentences   -> matched MLM control + RSR-on-sentences
#   B  rsr,     template         -> RSR-on-templates (the original rsr_all)
# The control does not use the RSR sampler, so it is run once per seed in A
# rather than duplicated in B.
#
# RESUMABLE. Two power-offs cost most of a night, so every arm checks for its own
# finished output first and skips if present. Re-running this script after any
# interruption picks up where it stopped rather than starting over.
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS=2000
EVAL_EVERY=100
CONTEXTS="sentences,bare,template"
SEEDS="1 2 3"
# The script's default pool is artifacts/sentences, which is the SIMPLE ENGLISH
# corpus. Every current result sits on the strict full-Wikipedia pool, so this
# has to be passed explicitly or the run compares against nothing we have.
POOL=artifacts/sentences_strict
SCRIPT=experiments/rsr_pretraining/bert_rsr_continued_pretraining.py
LOG=logs
mkdir -p "$LOG"

# done_already <regime> <tag> : is there a finished CSV for this arm?
done_already() {
  ls results/bert_continued_"$1"_"$2"_*.csv >/dev/null 2>&1
}

for seed in $SEEDS; do
  tag="ctx_sentences_s${seed}"
  if done_already rsr "$tag" && done_already baseline "$tag"; then
    echo "=== seed $seed : sentences arm already complete, skipping ==="
  else
    echo "=== seed $seed : control + RSR on real sentences ==="
    python "$SCRIPT" --regime compare --combine weighted --lambda 0.7 \
      --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$seed" \
      --rsr-context sentences --eval-contexts "$CONTEXTS" --sentences-dir "$POOL" \
      --save-tag "$tag" \
      > "$LOG/${tag}.log" 2>&1
    echo "  exit $?"
  fi

  tag="ctx_template_s${seed}"
  if done_already rsr "$tag"; then
    echo "=== seed $seed : template arm already complete, skipping ==="
  else
    echo "=== seed $seed : RSR on templates ==="
    python "$SCRIPT" --regime rsr --combine weighted --lambda 0.7 \
      --steps "$STEPS" --eval-every "$EVAL_EVERY" --seed "$seed" \
      --rsr-context template --eval-contexts "$CONTEXTS" --sentences-dir "$POOL" \
      --save-tag "$tag" \
      > "$LOG/${tag}.log" 2>&1
    echo "  exit $?"
  fi
done

echo "done. results in results/, logs in $LOG"
