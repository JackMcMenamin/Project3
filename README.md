# Project 3: Representational Similarity Regularisation (RSR)

Experiment 3 of the PhD. RSR adds a second objective to a pre-trained language
model: make the model's word-pair similarities rank the way human similarity
judgements do. The loss is a differentiable Spearman correlation,
`L_rsr = 1 - rho_soft(model_sims, human_sims)`, over the cosine similarities of
the model's contextual word vectors. Supervision is pooled from SimVerb-3500 and
THINGS (2,652 words; MEN is deliberately excluded because it rates relatedness
rather than similarity). Evaluation is held-out SimLex-999, partitioned by how
many of each pair's words appeared anywhere in the supervision:

| partition | n | meaning |
|---|---|---|
| Both | 294 | both words had human ratings during training |
| One | 215 | one did |
| Neither | 480 | neither did; this is the generalisation test |

The running notes, in the order things were
found, are `RSR_July/notes/findings.md`.

---

## What the results say

**RSR improves the pairs it was never supervised on, and not the ones it was.**
Continuing BERT's pre-training with `0.3 * MLM + 0.7 * RSR` for 800 steps:

| | Neither | Both | seeds |
|---|---|---|---|
| SimLex-999, vs matched MLM control | **+0.070** (p = 0.0001) | -0.009 (p = 0.46) | 5 |
| MEN (never used in training or model selection) | **+0.062** | +0.027 | 3 |
| Pereira fMRI, RSA against 10 subjects | **+0.049** (significant in every seed) | -0.028 | 3 |

GLUE is unchanged across MRPC, RTE, CoLA, STS-B and SST-2 (105 fine-tuning runs).

**The same signature appears in GPT-2**, with causal LM as the anchor instead of
MLM and the read-out at layer 6 (GPT-2's final layer is degenerate for this:
every cosine sits at 0.994 +/- 0.006). Three seeds: SimLex Neither **+0.026**,
Both -0.052; MEN Neither **+0.051**.

**Post-processing methods cannot do this.** On identical supervision,
retrofitting improves the supervised pairs by +0.190 and leaves the unsupervised
ones untouched (exactly 0.000, arithmetically). Post-specialisation reaches the
unsupervised pairs and makes them worse (-0.026). RSR gives +0.091 against the
same reference.

Things that were checked and did not help, or turned out not to matter: blending
layers (significantly worse), decaying the RSR weight over training, morphological
leakage between the partitions (changes the result by 0.0004), early stopping
(the Neither gain survives without it).

---

## Repository layout

```
RSR_July/notes/findings.md      running research notes with all the raw numbers

experiments/
  contextual_finetune/          the current pipeline
    bert_rsr_continued_pretraining.py   BERT: MLM + RSR continued pre-training
    gpt2_rsr_continued_pretraining.py   GPT-2 port; imports the BERT module so the
                                        supervision, partition, corpus and loss are
                                        the same code
    eval_men_transfer.py                MEN as a second benchmark
    eval_retrofit_baseline.py           retrofitting on the same supervision
    eval_postspec_baseline.py           post-specialisation on the same supervision
    eval_strict_neither.py              morphology contamination check
    aggregate_seeds.py                  seed aggregation + significance
    plot_training_curves.py             per-partition trajectories
    plot_context_comparison.py          template vs sentence training, three eval contexts
  runners/                      batch scripts for the multi-seed runs (.sh / .ps1)
  contextual/                   sentence harvesting (harvest_sentences_strict.py,
                                filter_sentence_pool.py) and the earlier cached-vector
                                pipeline it grew out of
  word2vec/                     the original word2vec pilot

RSR_July/                       RoBERTa replication of the earlier real-sentence
                                experiment, plus brain RSA and GLUE for our BERT
                                checkpoints (own README)
  results/replication/  brain/  downstream/

src/rsr/                        shared library for the word-level and cached-vector
                                experiments (not used by contextual_finetune/)
docs/                           differentiation report, older decks
figures/                        figures from earlier stages
archive/                        superseded code, kept for provenance (own README)

data/  artifacts/  models/  results/    gitignored: corpora, harvested sentences,
                                        checkpoints, per-run CSVs
```

---

## Where to look

In rough order of usefulness:

2. `RSR_July/notes/findings.md`. The notes, chronological. Each section is one
   experiment with the numbers, the run IDs, and any correction to an earlier
   claim.
3. `experiments/contextual_finetune/bert_rsr_continued_pretraining.py`. The
   method. The `Config` class and `rsr_loss_for_batch` are the parts that matter;
   the rest is plumbing.
4. `experiments/contextual_finetune/gpt2_rsr_continued_pretraining.py`. The
   header explains what had to change for a causal model and why the read-out
   layer differs.
5. `figures/`. Layer probe, layer-arm comparison, extended training, and
   the template-vs-sentence comparison.
6. `RSR_July/results/brain/` and `RSR_July/results/downstream/`. The fMRI and
   GLUE numbers, three seeds each.

---

## Running things

The training corpus must be passed explicitly. The script's default pool is
Simple English Wikipedia; every current result is on the full-Wikipedia strict
pool, and the two are not comparable.

```bash
# BERT, the headline configuration: matched MLM control + RSR, one seed
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py \
    --regime compare --combine weighted --lambda 0.7 --steps 800 \
    --sentences-dir artifacts/sentences_strict --seed 1

# GPT-2, same recipe; layer 6 because the final layer is unusable (see header)
python experiments/contextual_finetune/gpt2_rsr_continued_pretraining.py \
    --regime compare --lambda 0.7 --rsr-layer 6 \
    --sentences-dir artifacts/sentences_strict --seed 1

# which layer of the untouched model is already best at SimLex (no training)
python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --probe-layers
python experiments/contextual_finetune/gpt2_rsr_continued_pretraining.py --probe-layers \
    --sentences-dir artifacts/sentences_strict

# baselines, evaluation only, no training
python experiments/contextual_finetune/eval_retrofit_baseline.py
python experiments/contextual_finetune/eval_postspec_baseline.py
python experiments/contextual_finetune/eval_men_transfer.py
```

Multi-seed runs go through `experiments/runners/`. They are resumable: each arm
checks for its own finished output and skips it, so an interrupted batch picks up
where it stopped.

Brain RSA and GLUE live in `RSR_July/`; see its README.

### Setup

```bash
pip install torch transformers numpy pandas scipy scikit-learn tqdm matplotlib
pip install torchsort            # differentiable ranking; falls back to a slower path without it
```

The sentence pool is built once from the local enwiki dump with
`experiments/contextual/harvest_sentences_strict.py` followed by
`filter_sentence_pool.py`, and lands in `artifacts/sentences_strict/`. Slow; do
not rerun unless the target inventory changes.

---

## Protocol

Reported numbers are means over seeds against a matched control that differs
from the RSR arm in exactly one thing, the RSR term. SimLex headline: five seeds.
Everything else: three. Nothing rests on a single run. Early stopping picks the
best all-pairs checkpoint; the notes report what happens without it.

Two practical points that cost time to learn. Run-to-run divergence at a fixed
seed grows with training length (0.0002 at step 400, 0.015 by step 2000), so
long-horizon claims need several seeds before they are quoted. And every
evaluation script that stands alone must set `SENTENCES_DIR` to the strict pool
itself, because the module default is the wrong corpus.
