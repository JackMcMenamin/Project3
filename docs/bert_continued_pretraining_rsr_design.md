# Design note — BERT continued pre-training with RSR (sentence-level)

**Status:** proposal, pre-implementation. Written to take back to supervisor before any training code is written.
**Scope:** BERT only for now (per supervisor). GPT-2 can follow the same template later.
**Date:** 2026-06-01

---

## 1. The problem this solves

The current contextual pipeline (`experiments/contextual/`) embeds each target
word inside real sentences, mean-pools the target's subword hidden states, and
averages across occurrences into **one cached vector per word**. BERT is run
**once, frozen, offline**; only a projection head trains on the cached vectors.

**Supervisor's criticism:** BERT itself never learns anything — we are just
*gleaning* static embeddings from BERT's existing contextual outputs. The
sentences are used once to extract a fixed representation; BERT's weights are
untouched. Conceptually this is "use BERT as a fancy static encoder," which is
the same critique the word-level approach drew, just with sentences instead of
isolated words.

**Fix:** keep BERT's own weights training (**continued pre-training**), so the
similarity constraint reshapes BERT's representations directly — while a
masked-language-modelling (MLM) signal keeps BERT a functioning language model.

This is a direct adaptation of Mark Ormerod's RSR framework (thesis Ch. 7,
`Mark.txt`), changing the primary task from STS to MLM and the RDM items from
whole sentences to in-context target-word tokens.

---

## 2. Two training regimes

### Regime 1 — baseline: plain MLM (control)
Continue pre-training BERT the normal way: mask tokens, predict them,
cross-entropy loss. Any sentences. **No RSR.** This is the control the RSR
regime is compared against — "what does continued pre-training alone do?"

### Regime 2 — interleaved MLM + RSR (the experiment)
Alternate two batch types (mirrors the Word2Vec interleaving in
`experiments/word2vec/run_seeds_interleaved.py`):

- **(a) MLM batch** — standard masked-token prediction, cross-entropy. Any
  sentences. Preserves BERT's language ability / guards against catastrophic
  forgetting while the similarity objective pushes the geometry.

- **(b) RSR batch** — N sentences, **each selected because it contains one of
  our target words** (aardvark, chipmunk, …):
  1. Forward the batch through BERT (**gradients on** — BERT is training).
  2. For each sentence, extract the hidden state(s) of its target word token(s)
     and pool to one vector (pooling choice: §5).
  3. Build the model RDM over the N target vectors: cosine similarity for each
     pair, upper triangle excluding diagonal → **N(N−1)/2** values.
  4. Build the ground-truth RDM from the human similarity scores for those same
     N word pairs.
  5. Loss `R_RS = 1 − ρ_soft(model_RDM, target_RDM)` — differentiable
     (soft) Spearman, via `torchsort` (same loss already in `src/rsr/losses.py`).

This is Mark's RSR term exactly (Ch. 7, Eq. 1), with the RDM built from
**in-context word tokens** rather than sentence embeddings.

### Combining the two signals
Two options, to decide with supervisor:
- **Interleaved** (supervisor's phrasing): separate batches, alternate them.
  Each batch contributes its own loss/step. Matches the W2V experiment.
- **Weighted** (Mark's phrasing, Ch. 7 Eq. 2): one batch, combined loss
  `L = (1−λ)·L_MLM + λ·R_RS`.

Interleaved is the stated request; note the equivalence so we can ablate.

---

## 3. Data flow

- **Supervision (ground-truth RDM):** MEN + SimVerb-3500 + THINGS, pooled and
  normalised — the existing `rsr.datasets.load_all_rsr_datasets()`.
- **Evaluation:** SimLex-999, **held out** (never used in training). Confirmed
  with supervisor 2026-06-01 — he wrote "SemLex-999" but meant the supervision
  pool stays MEN/SimVerb/THINGS and SimLex remains eval-only. Using SimLex as
  the training target would contaminate evaluation.
- **RSR-batch sentences:** the harvested sentences already cached in
  `artifacts/sentences/<slug>.jsonl` (Stage A of the contextual pipeline) can be
  reused — but now fed through BERT live with gradients, not pre-embedded.
- **MLM-batch sentences:** any reasonable sentence source (the same harvest
  corpus, or a generic Wikipedia sample).

---

## 4. Freezing & hyperparameters (inherited from Mark, Ch. 7)

**Recommended starting point — keep Mark's scheme**, because it is the validated
prior-art default, matches the current paper's transformer setup for continuity,
and is the conservative choice against catastrophic forgetting (which matters
*more* here, since MLM+RSR is a more aggressive intervention than Mark's
STS+RSR):

| Setting | Value | Source / rationale |
|---|---|---|
| Frozen | BERT embeddings + encoder layers 0–10 | Mark's `Main_Experiments.ipynb`: `for module in [embeddings, encoder.layer[:11]]: requires_grad = False` |
| Trainable | final encoder block (layer 11) + projection head | isolates the causal claim; limits capacity |
| Projection dim | 128 | Ch. 7.3.1; similarity easier to enforce in low-dim space |
| RSR batch size N | 5 (→10 pairwise sims) | Ch. 7.3.5; trade-off: larger N = richer comparisons but dilutes signal, memory ∝ N² |
| Reg. strength λ | 0.9 | Ch. 7.4.2 found 0.9 > 0.1 on average (his default was 0.1) |
| Learning rate | 1e-4, Adam | Ch. 7.4.5; higher (1e-2) hurt |
| Soft-rank | torchsort (Blondel et al. 2020) | Ch. 7.3.6; already in `src/rsr/losses.py` |

**Why freeze rather than full fine-tune** (the question Jack asked):
1. **Isolates the claim** — any geometry change is attributable to targeted RSR
   pressure, not wholesale retraining (Ch. 7.3.1 / paper §6.1).
2. **Prevents catastrophic forgetting** — limited trainable capacity + low LR
   stops the similarity objective from overwriting BERT's pretrained linguistic
   knowledge (Ch. 7.4.3, 7.4.7 discuss this risk explicitly).
3. **Projection head is where the constraint lives** — a compact space the RSR
   loss can actually shape (Ch. 7.3.1).

**Ablation to consider later:** unfreeze more layers / vary projection dim
(Mark's hidden-size experiment, Ch. 7.4.3, is exactly this kind of capacity
sweep). Run only if the frozen-default results justify it.

---

## 5. Design choices (all settled 2026-06-01 — each a one-line Config change)

1. **MLM vs RSR mix.** Interleave as separate batches (supervisor's request, not
   Mark's single-batch weighting), **RSR-majority: 1 MLM : 2 RSR per cycle**.
   Rationale: the MLM batches only need to keep BERT a working language model;
   Mark leaned heavily on similarity (λ=0.9), so we bias the interleave the same
   way. `Config.interleave_cycle=3, mlm_per_cycle=1`.
2. **What the RDM compares.** The **target word's token in context** (e.g.
   "aardvark" inside its sentence), mean-pooled over its subwords — *not* a
   whole-sentence vector. This is the key adaptation of Mark's
   sentence-vs-sentence RDM, and matches the supervisor's "similarity of target
   word tokens within a batch."
3. **RSR batch assembly.** Option (b): pick any target words; build the
   soft-Spearman correlation only over pairs that have a human score, mask the
   rest. Less restrictive; revisit (a) "fully-connected batches" only if batches
   come out too sparse to give a useful gradient.
4. **MLM sentence source.** Reuse the harvested target-word sentences — the
   supervisor said MLM "could be on any sentences", and these qualify, so no
   extra data is needed. Standard 15% masking.
5. **Multi-subword target pooling.** Mean-pool subwords (matches the existing
   pipeline and Mark).
6. **Eval encoding.** Same in-context extraction as training (sentences → BERT →
   pooled target token), for an honest comparison. SimLex held out.
7. **Polysemy / sense.** Ignored (frequency-mixed), as in the rest of the
   project — fine for parity with W2V and because SimLex isn't
   sense-disambiguated.

---

## 6. How this maps onto the existing repo

- Reuse `src/rsr/losses.py` (soft-Spearman) and `rsr.datasets` (ground-truth
  pairs) unchanged.
- Reuse `artifacts/sentences/` as the RSR-batch sentence source (fed live, not
  pre-embedded).
- New code lives in a new `experiments/contextual_finetune/` (or similar) — it
  is a *different* experiment from the frozen-vector `experiments/contextual/`
  pipeline, so keep them side by side rather than overwriting.
- The frozen-cached-vector pipeline (`experiments/contextual/`) becomes the
  "static-extraction" comparison point; this new regime is the
  "BERT-actually-trains" version.

---

## 7. One thing to flag to supervisor

Mark's RDM items were **sentences**; here they are **in-context word tokens**.
That is the key adaptation and the answer to "you're just gleaning embeddings":
BERT trains on real sentences (MLM), and the similarity constraint operates on
the contextualised word representations that the behavioural data actually
rates. Worth confirming he agrees the RDM is built over target-word tokens
(not, say, the [CLS]/sentence vector).
