# Contextual fine-tuning — BERT continued pre-training with RSR

The current direction: **BERT's own weights train** so the similarity constraint
reshapes its representations directly, instead of just reading off frozen
embeddings (the criticism of `../contextual/`). Implements
`docs/bert_continued_pretraining_rsr_design.md`.

## File

- **`bert_rsr_continued_pretraining.py`** — one self-contained script (no project
  imports) so it can be handed to / run by the supervisor. BERT only for now;
  GPT-2 will follow the same template.

## First: find a working recipe (`--sweep`)

The first full run **degraded** SimLex (RSR all 0.45 → 0.31) — the N=5 RSR batch
gave a noisy 10-pair Spearman gradient and the 1:2 RSR-majority interleave
starved BERT's language signal (2 of every 3 batches had no MLM). Before a full
run, sweep a few recipes at reduced steps to find one that actually *lifts*
SimLex:

```bash
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --sweep
```

It loads data once, then tries several (RSR batch size, MLM:RSR interleave, lr)
combos for ~600 steps each and prints which best improves held-out SimLex over
vanilla. Pick the winning recipe, set it in `Config`, then do the full
comparison run below. (Defaults have already been moved to N=24, balanced 1:1,
lr 5e-5 — the sweep confirms / refines that.)

## Default: full comparison

Running with no flags trains **vanilla BERT → RSR BERT** and **vanilla → MLM-only**
(each from its own fresh model), then prints a paper-Table-2-style comparison —
Vanilla / RSR / Delta, partitioned by both/one/neither-in-RSR overlap, plus a
column isolating the RSR gain over plain continued training.

```bash
# default = the full comparison table:
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py
```

| `--regime`  | What it does |
|-------------|--------------|
| `compare` (default) | Run both regimes and print the comparison table. |
| `rsr`       | **The experiment only.** Interleaved batches: MLM (cross-entropy on masked tokens) and RSR (align the contextual target-word similarity RDM to the human RDM via soft-Spearman). |
| `baseline`  | **The control only.** Continued MLM, no RSR. |

```bash
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --regime rsr
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --regime baseline

# fast wiring check (~20 steps each, CPU-friendly):
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --smoke
```

## What it reuses

- Human similarity: MEN + SimVerb-3500 + THINGS (pooled, normalised). SimLex-999
  held out for eval, partitioned by both/one/neither-in-RSR overlap.
- Sentences: the already-harvested `artifacts/sentences/<slug>.jsonl` (up to 50
  real Wikipedia sentences per target word), fed through BERT live with
  gradients — and reused as the MLM sentence pool.

## Key defaults (from Mark Ormerod's RSR chapter — see design note)

Freeze embeddings + encoder layers 0–10, train only the final block + a 128-d
projection head; lr 1e-4, RSR batch N=5, soft-Spearman via torchsort (optional).
RSR batches use **Option B**: any target words, loss masked to the pairs that
have a human score.

Output: `results/bert_continued_{regime}_{timestamp}.csv` (SimLex rho per
partition across training steps).
