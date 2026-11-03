#!/usr/bin/env bash
# Three GPT-2 follow-ups, queued in order of value.
#
# 1. EXTENDED TRAINING, 8000 steps, 3 seeds. The BERT half's most-cited figure
#    is the long run: the unsupervised gain persists while all-pairs decays to
#    zero and the supervised pairs fall below control. GPT-2 has nothing past
#    2000 steps, and a sharper version of the question, since its Both partition
#    was already at -0.088 by step 2000 without early stopping. Early stopping is
#    OFF here: on GPT-2 the all-pairs score declines, so a best-all-pairs rule
#    returns the untrained model, and matched-step curves are what get read.
#    1e-5 matches the BERT extended run.
#
# 2. TEMPLATES VS REAL SENTENCES, 3 seeds, scored three ways. Exists for BERT
#    only. Decides whether "bare-word scoring cannot tell good training from
#    bad" is a BERT fact or a general one. Same design as the BERT runner: the
#    control does not use the RSR sampler, so it is run once per seed in the
#    sentences arm and not duplicated.
#
# 3. LAYER 12, 3 seeds. Turns the BERT layer finding into a prediction. On BERT
#    a better starting layer gained less. GPT-2's gap between layers is far
#    wider (0.161 at L12 vs 0.529 at L6), so if the pattern holds L12 should
#    gain MORE. Same recipe as the main runs, one flag different.
#
# RESUMABLE: every arm checks for its own finished output and skips if present.
set -u
cd "$(dirname "$0")/../.." || exit 1

SEEDS="1 2 3"
POOL=artifacts/sentences_strict
SCRIPT=experiments/rsr_pretraining/gpt2_rsr_continued_pretraining.py
LOG=logs
mkdir -p "$LOG"

have() { ls results/gpt2_continued_"$1"_"$2"_*.csv >/dev/null 2>&1; }

# ---- 1. extended training -------------------------------------------------
for seed in $SEEDS; do
  tag="long_s${seed}"
  if have rsr "$tag" && have baseline "$tag"; then
    echo "=== long seed $seed : already complete, skipping ==="; continue
  fi
  echo "=== long seed $seed : 8000 steps @1e-5, no early stop, control + RSR ==="
  python "$SCRIPT" --regime compare --lambda 0.7 --rsr-layer 6 \
    --steps 8000 --eval-every 400 --lr 1e-5 --no-early-stop --seed "$seed" \
    --sentences-dir "$POOL" --save-tag "$tag" \
    > "$LOG/gpt2_${tag}.log" 2>&1
  echo "  exit $?"
done

# ---- 2. templates vs sentences, three eval contexts -----------------------
CTX="sentences,bare,template"
for seed in $SEEDS; do
  tag="ctx_sentences_s${seed}"
  if have rsr "$tag" && have baseline "$tag"; then
    echo "=== ctx sentences seed $seed : already complete, skipping ==="
  else
    echo "=== ctx sentences seed $seed : control + RSR on real sentences ==="
    python "$SCRIPT" --regime compare --lambda 0.7 --rsr-layer 6 \
      --steps 2000 --eval-every 200 --seed "$seed" \
      --rsr-context sentences --eval-contexts "$CTX" \
      --sentences-dir "$POOL" --save-tag "$tag" \
      > "$LOG/gpt2_${tag}.log" 2>&1
    echo "  exit $?"
  fi
  tag="ctx_template_s${seed}"
  if have rsr "$tag"; then
    echo "=== ctx template seed $seed : already complete, skipping ==="
  else
    echo "=== ctx template seed $seed : RSR on templates ==="
    python "$SCRIPT" --regime rsr --lambda 0.7 --rsr-layer 6 \
      --steps 2000 --eval-every 200 --seed "$seed" \
      --rsr-context template --eval-contexts "$CTX" \
      --sentences-dir "$POOL" --save-tag "$tag" \
      > "$LOG/gpt2_${tag}.log" 2>&1
    echo "  exit $?"
  fi
done

# ---- 3. layer 12 ----------------------------------------------------------
for seed in $SEEDS; do
  tag="L12_s${seed}"
  if have rsr "$tag" && have baseline "$tag"; then
    echo "=== L12 seed $seed : already complete, skipping ==="; continue
  fi
  echo "=== L12 seed $seed : control + RSR, read-out at layer 12 ==="
  python "$SCRIPT" --regime compare --lambda 0.7 --rsr-layer 12 \
    --steps 2000 --eval-every 200 --seed "$seed" \
    --sentences-dir "$POOL" --save-tag "$tag" \
    > "$LOG/gpt2_${tag}.log" 2>&1
  echo "  exit $?"
done

echo "done. results in results/gpt2_continued_*, logs in $LOG"
