# RSR: technical report

A running, structured record of the method, every experiment run, why it was
run, what it showed, and which earlier claims it changed. `findings.md` stays
as the working notebook with the full discussion; this file is the reference
version, kept current and written to be lifted into a paper.

Last updated: 26 November 2026.

How this file is maintained:

- Section 2 (current state) is edited in place whenever the headline changes.
- Section 4 (experiment log) is append-only. Each entry records the question,
  the configuration, the result and the conclusion. If a later experiment
  overturns an entry, the entry stays and gets a note pointing forward; the
  correction itself goes in section 5.
- Numbers are SimLex-999 Spearman rho, RSR arm minus its matched control, on
  the Neither partition, unless stated otherwise. Seeds and the checkpoint rule
  (endpoint or early-stopped) are given with every number.

Contents:

1. Method
2. Current state
3. Protocol and statistics
4. Experiment log
5. Corrections and retracted claims
6. Open questions and planned work
7. Reproduction

---

## 1. Method

### 1.1 Objective

Continued pre-training of a pre-trained language model with an auxiliary
representational similarity term. Per step:

    L = (1 - lambda) * L_anchor + lambda * L_RSR,   lambda = 0.7

- `L_anchor` is the model's own pre-training loss: masked LM for BERT
  (masking probability 0.15), causal LM for GPT-2.
- `L_RSR = 1 - rho_soft(c, h)`, where `c` is the vector of cosine
  similarities between the model's contextual vectors for every scored word
  pair in the batch, `h` the corresponding human ratings scaled to [0, 1], and
  `rho_soft` a differentiable rank correlation.
- Both terms are combined every step ("weighted"). The original RoBERTa
  pipeline alternates them instead; the weighted form at lambda 0.7 was
  selected in early sweeps (details to be backfilled, see section 6).
- The whole model trains. Gradient norm clipped to 1.0. BERT uses Adam,
  GPT-2 AdamW, learning rate 5e-5 unless stated.

**The soft rank is effectively Pearson.** torchsort is not installed, so
every run used the sigmoid fallback (`rsr_core/loss.py`): the rank of x_i is
replaced by `0.5 + sum_j sigmoid(strength * (x_i - x_j))` at strength 1.0.
Cosines and scaled ratings differ by small amounts, so the sigmoid operates in
its linear region and the soft ranks are close to an affine map of the raw
values. On simulated batches `rho_soft` matches Pearson to within 0.006 and
differs from true Spearman by up to 0.12. The method as run is a Pearson
correlation loss on cosines and should be described that way unless rerun
with sharper ranks.

### 1.2 Word vectors

- A word's vector in a sentence is the mean of the hidden states of its
  sub-word pieces at the read-out layer (BERT layer 12; GPT-2 layer 6, see
  E13). Span location verified on GPT-2 at 100% over 300 sampled words.
- Training: one sentence per word, redrawn every batch, from the harvested
  pool (or the fixed frame "The word is X." in template arms).
- Evaluation: the mean over up to 20 harvested sentences per word, then
  cosine. "Bare" (word alone) and "template" evaluation contexts are also
  logged in some arms; see E12 for why they are not used as effect sizes.

### 1.3 Supervision

| source | pairs | words | rating |
|---|---|---|---|
| SimVerb-3500 | 3,500 (3,492 after duplicates and overlaps) | 827 verbs | human similarity, SimLex-style instructions |
| THINGS (SPoSE 66-d) | every pair of 1,854 concepts, 1,717,731 | 1,854 nouns | cosine of SPoSE embeddings, from triplet odd-one-out judgements on object images |

Combined lookup: 1,721,223 scored pairs over 2,652 words that have sentences.
MEN is held out entirely and used only for transfer evaluation. WordSim-353 is
not used (31% sentence coverage).

Optional transforms (all off in the reported runs before November 2026):

- `--exclude-simlex-pairs`: drop every scored pair that is also a SimLex pair,
  in either order, including inflected copies. Removes 274 pairs and no words.
- `--inflect-verbs`: add third person, past and gerund copies of each SimVerb
  pair (lemminflect, last listed form, same score). An inflected copy never
  overrides an existing pair. 2,565 new forms; harvested from enwiki, 2,495
  reach 10 sentences.
- `--things-pairs N`: replace all THINGS pairs with a fixed random sample of N
  (seed 42, as in the original pipeline). N = 10,000 drops one word.

### 1.4 RSR batches

- `--sampler words` (reported runs): draw 24 words uniformly from targets with
  sentences; every scored pair among them enters the loss. About 139 scored
  pairs per batch, 0.26 of them SimVerb.
- `--sampler pairs`: draw scored pairs and take their words until 24 are
  placed; incidental scored pairs among those words also count. Optionally
  `--simverb-share F` draws a fraction F of the pairs from SimVerb.
- A batch with fewer than 8 scored pairs is skipped.

Scored pairs per batch (audit, 1,000 simulated batches, and measured in
training):

| configuration | scored pairs | SimVerb pairs | SimVerb share |
|---|---|---|---|
| words, all THINGS (reported) | 139 | 0.26 | 0.2% |
| pairs, all THINGS | 275 | 0.03 | 0.0% |
| pairs, all THINGS, share 0.33 | 134 | 4.6 | 3.4% |
| pairs, 10k THINGS | 13 | 3.4 | 26% |
| pairs, 10k THINGS, inflected | 12.6 | 7.4 (1.9 base + 5.5 inflected) | 59% |

### 1.5 Anchor corpus and sentence pool

- Pool: `artifacts/sentences_strict`, harvested from the full English
  Wikipedia dump with the original pipeline's rules: target as a lowercase
  token, 5 to 100 words, terminal punctuation, at most one sentence per article
  per target, up to 50 per word. Targets that only occur capitalised fall back
  to a case-insensitive pool (11 targets), recorded in the manifest.
- The anchor (MLM/CLM) corpus is the harvested sentences of 400 random
  supervised words, about 19,500 sentences, fixed per seed and shared by both
  arms of a comparison.
- `--split-sentences`: training (anchor and RSR) uses the first half of each
  word's sentences, evaluation the second half.

### 1.6 Evaluation

SimLex-999, partitioned by exposure to the supervision vocabulary:

| partition | definition | pairs | of which N / V / A |
|---|---|---|---|
| Both | both words supervised | 297 | 105 / 192 / 0 |
| One | exactly one supervised | 219 | 191 / 28 / 0 |
| Neither | neither supervised | 483 | 370 / 2 / 111 |

989 of 999 pairs have sentences for both words and are scored. Neither is the
pre-specified test: those words never appear in any RSR batch, so a gain there
is a change to words the objective never touched. Since November 2026 every
run also logs Both split into trained/unseen pairs and every partition by part
of speech (cells under 10 pairs are not printed).

The partitions are confounded with part of speech: Both is two thirds verbs,
Neither almost entirely nouns and adjectives. Any statement about Both is
largely a statement about verbs.

Other evaluations: MEN (relatedness, 2,980 of 3,000 pairs covered, same
partition), Pereira et al. (2018) fMRI RSA over 180 concepts and 10 subjects,
GLUE (MRPC, RTE, CoLA, STS-B, SST-2).

---

## 2. Current state

**Recommended configuration (BERT):** weighted lambda 0.7, layer 12, lr 5e-5,
2,000 steps, endpoint model, SimLex pairs excluded, split sentences, pair-first
sampling over a 10k THINGS sample, inflected SimVerb pairs.

BERT, 2,000 steps, endpoint, 5 seeds, RSR minus control:

| configuration | All | Both | One | Neither | Neither range |
|---|---|---|---|---|---|
| reported sampling | -0.016 | -0.066 | -0.044 | +0.045 | [+0.033, +0.066] |
| + SimLex excluded, split sentences | -0.019 | -0.053 | -0.039 | +0.039 | [+0.017, +0.060] |
| + pair-first, 10k THINGS | +0.123 | +0.212 | +0.019 | +0.155 | [+0.137, +0.183] |
| + inflected SimVerb | **+0.144** | **+0.219** | +0.025 | **+0.188** | [+0.179, +0.200] |

Earlier headline numbers, for reference, all on the reported sampling:

| model | setting | seeds | SimLex Neither | MEN Neither |
|---|---|---|---|---|
| BERT | early-stopped | 5 | +0.070 | +0.062 |
| BERT | 8,000 steps at 1e-5, endpoint | 3 | +0.065 | |
| GPT-2, layer 6 | 2,000 steps, early-stopped | 3 | +0.026 | +0.051 |
| GPT-2, layer 6 | 2,000 steps, endpoint | 3 | +0.021 | |
| GPT-2, layer 6 | 8,000 steps at 1e-5, endpoint | 3 | +0.023 | +0.060 |

What currently stands:

1. RSR improves SimLex on words it never received ratings for, in BERT and
   GPT-2, with and without early stopping.
2. Under the reported sampling the supervised pairs get worse and the effect is
   modest (+0.045 endpoint). Verbs were supervised in name only.
3. With SimVerb pairs actually present in batches, the gain on unsupervised
   words roughly quadruples, the supervised pairs improve strongly, and All
   pairs rises by +0.14. Unsupervised adjectives gain +0.22.
4. The gain follows the number of SimVerb pairs per batch: little below one,
   most of it by about five, and still rising slowly at ten (E20, E24, E27).
   The THINGS cut is not needed for it.
5. Nouns supervised by THINGS get slightly worse on SimLex in every
   configuration.
6. Out-of-context evaluation (bare word, null template) misstates the effect
   and is not used for effect sizes.

Other evaluations of the recommended configuration (RSR minus control,
partitions from that run's own vocabulary):

| evaluation | seeds | All | Both | One | Neither |
|---|---|---|---|---|---|
| MEN | 5 | +0.040 | +0.019 | +0.055 | +0.032 |
| MEN, reported sampling (for comparison) | 5 | +0.071 | +0.047 | +0.083 | +0.063 |
| Pereira RSA, layer 12 | 3 x 10 subjects | +0.026 | -0.004 | +0.008 | +0.027 |
| Pereira RSA, layer 7 | 3 x 10 subjects | +0.028 | +0.001 | +0.006 | +0.032 |
| SimLex Neither, related forms removed | 5 | | | | +0.186 |
| GPT-2 layer 6, SimLex | 3 | +0.064 | +0.117 | +0.032 | +0.078 |
| GPT-2 layer 6, MEN | 3 | +0.053 | +0.042 | +0.063 | +0.050 |

7. The SimLex gain from the recommended configuration is not morphological
   leakage from the inflected forms (E22).
8. It transfers to MEN and to brain RSA, but the MEN transfer is about half
   that of the reported sampling, while the SimLex gain is four times larger
   (E21, E23). This is not a similarity for relatedness trade: with all of
   THINGS kept, MEN holds as SimVerb rises (E27), and GPT-2 keeps its MEN
   gain (E25). The MEN cost comes from the inflected SimVerb forms; the 10k
   THINGS sample without them keeps the full MEN transfer (+0.063, E29).
9. GPT-2 under the recommended configuration shows the same shape at about
   40% of BERT's SimLex size (Neither +0.078), with a larger anchor cost (E25).

10. GLUE is at parity on four of five tasks; MRPC loses 0.022 in every
    checkpoint (E26). With all of THINGS kept (share33) the MRPC loss is
    -0.007 and not consistent (E28).

11. A candidate alternative to the recommended configuration: pair-first
    sampling over all of THINGS with two thirds of draws from SimVerb
    (share66). SimLex Neither +0.160, All +0.110, MEN Neither +0.065, 3
    seeds (E27). GLUE not yet run on it.

| arm, all of THINGS, 3 seeds | SimVerb pairs per batch | SimLex All | SimLex Neither | MEN Neither | MRPC |
|---|---|---|---|---|---|
| pairs_full | 0.03 | -0.026 | +0.031 | +0.051 | |
| share10 | 1.3 | -0.003 | +0.048 | +0.055 | |
| share33 | 4.6 | +0.086 | +0.131 | +0.064 | -0.007 |
| share66 | 10.5 | +0.110 | +0.160 | +0.065 | |
| pairs (10k THINGS, 5 seeds) | 3.4 | +0.123 | +0.155 | +0.063 | |
| recommended (10k THINGS, inflected, 5 seeds) | 7.4 | +0.144 | +0.188 | +0.032 | -0.022 |

Every evaluation of the earlier configuration now has a recommended-
configuration counterpart except the read-out and extended-training
experiments.

---

## 3. Protocol and statistics

- **Matched control.** Every RSR arm has a control from the same seed, same
  initial weights, same anchor corpus and schedule, with `lambda = 0` (anchor
  loss only). The control never calls the RSR sampler, so one control serves
  every RSR arm with the same seed and the same SimLex partition.
- **Checkpoint.** Endpoint model by default since November 2026.
  `--early-stop` keeps the best all-pairs SimLex checkpoint, which selects on
  the test set; runs before November 2026 marked "early-stopped" used it.
- **Seeds.** Reported numbers are means over seeds, with min/max ranges and
  one-sample or paired t-tests across seeds. Bootstrap intervals are planned.
- **Noise floor.** Two runs that share every input differ by about 0.003 at
  step 800 (GPU nondeterminism) and the divergence grows with training, to
  about 0.015 by step 2,000 at a fixed seed (E10). Single-seed numbers past
  step 1,000 are shapes, not values.
- **Known protocol issues.** The RSR arm and its control do not see identical
  anchor batches after step 1 (both draw from the global generators and the
  RSR arm draws more). Runs before the October 2026 refactor drew the anchor
  pool before seeding and are not exactly reproducible; both arms of a run
  still shared it.

---

## 4. Experiment log

Run files are in `results/` (not versioned); the timestamp in the file name
dates the run. Full discussion of each entry is in `findings.md` under the
heading given.

### E01. Reproduction of the original RoBERTa pipeline

- **Question.** Does the original code reproduce on this hardware before
  anything is built on it?
- **Setup.** Original code, two replicates, four modes, 8,000 steps, datasets
  regenerated by its own scripts.
- **Result.** Baseline 0.570 (original 0.557), vanilla_all 0.560 (0.549),
  vanilla_targetwords 0.571 (0.558), rsr_all with templates 0.627 (0.611),
  rsr_targetwords with real sentences 0.477 (0.486). fMRI pipeline: 2,296
  voxels, identical.
- **Conclusion.** Reproduces; the uniform +0.013 offset is the corpus (405k
  sentences against 336k). Ordering and conclusions identical.
- **Notes.** Real-sentence RSR is below baseline in the original pipeline;
  explaining why it works in this pipeline drove E06 to E09 and E12.
- findings: "The original RoBERTa results reproduce".

### E02. Early-stopped versus endpoint, short schedule

- **Question.** Is the result an artefact of keeping the best SimLex
  checkpoint?
- **Setup.** BERT, weighted 0.7, 800 steps, 3 seeds, no early stopping.
  Runs `20260803_*`.
- **Result.** Neither +0.044 (p = 0.030, positive every seed) against +0.053
  early-stopped on the same corpus. All pairs +0.012 (n.s.), Both -0.019.
- **Conclusion.** The Neither gain survives checkpoint choice; the all-pairs
  gain does not.
- findings: "The result holds without early stopping".

### E03. Sentence pool

- **Question.** How much does the harvested corpus matter?
- **Setup.** Three pools: original (Simple English Wikipedia, unfiltered),
  cleaned (same source, original filters applied afterwards), strict (full
  enwiki, original rules at harvest time). 5 seeds each, weighted 0.7, 800
  steps, early-stopped. Runs `20260817_*` to `20260818_*`.
- **Result.** Neither +0.053 / +0.037 / +0.070, all p < 0.001, 5 of 5 seeds in
  every pool. All, Both and One change sign or significance across pools.
- **Conclusion.** Only Neither is robust. RSR's absolute Neither score is
  nearly constant (0.614, 0.607, 0.606); the delta moves because vanilla BERT
  moves (0.550, 0.558, 0.515). Quote the corpus with the number. Strict is the
  pool from here on.
- findings: "The training corpus was doing more work than we thought".

### E04. Brain RSA and GLUE

- **Question.** Does the effect show up outside word-similarity benchmarks,
  and does RSR damage the model?
- **Setup.** Strict-pool checkpoints, layer 12, seed 1. Pereira RSA over 10
  subjects; GLUE, 5 tasks x 3 fine-tuning seeds, 10% warmup.
- **Result.** RSA, RSR minus control at layer 12: Neither +0.054 (p = 0.009,
  8/10 subjects), significant at 9 of 13 layers; All +0.046 (p = 0.033); Both
  -0.019 (n.s.). GLUE: RSR within seed noise of vanilla and control on all
  five tasks, no collapsed runs in 45.
- **Conclusion.** The unsupervised gain appears in brain data; no downstream
  damage.
- **Limit.** One training checkpoint; statistics are over subjects and
  fine-tuning seeds, not training runs.
- findings: "Brain and GLUE, re-run on the filtered corpus".

### E05. Layer read-out and layer weighting

- **Question.** Is reading the last layer the right choice, and does pooling
  layers or applying RSR at several layers help?
- **Setup.** Learned scalar mix, frozen probe-weighted mix, and separate RSR
  terms at layers 7, 9, 12 (`--rsr-loss-layers`), each 3 seeds, 800 steps,
  evaluated at layer 12. Separately, `--probe-layers` on the untrained model:
  single layer n against `AVG(L<=n)` (Vulic et al., 2020).
- **Result.** Neither: layer 12 +0.080, multiloss +0.070 (p = 0.29 against
  layer 12), learned mix +0.044 (p = 0.071), frozen mix +0.040 (p = 0.036).
  Probe: prefix averaging wins on All (0.542 at n = 9 against 0.523 for L8)
  but not on Neither (L9 0.554 against best prefix 0.551).
- **Conclusion.** No read-out tried improves the unsupervised gain. Several
  losses tie the last layer; averaging layers is worse. Layer 12 retained.
- findings: "Layer weighting, both readings of it"; "Layer read-out".

### E06. Decaying lambda

- **Question.** The gain peaks early and declines. Is the decline caused by
  sustained RSR pressure? (Motivated by the linear auxiliary-weight decay in
  McClure & Kriegeskorte, 2015.)
- **Setup.** `--lambda-decay linear` from 0.7 to 0, 3 seeds, layer 12, against
  constant 0.7. Runs `20260825_234026`, `20260826_003811`, `20260826_013329`.
- **Result.** Best checkpoint +0.072 against +0.079 (p = 0.17); endpoint
  +0.060 against +0.072 (p = 0.37). Over the last 400 steps the RSR arm loses
  0.0157 with lambda falling to 0 and 0.0162 at constant lambda.
- **Conclusion.** Not RSR overtraining: the decline continues at the same rate
  with no RSR gradient. Continued anchor training erodes the installed
  structure. The lever is training length or schedule, not lambda.
- findings: "Decaying the RSR weight does not help".

### E07. Learning rate 1e-5

- **Question.** Is the early peak an artefact of a high learning rate (5e-5
  against the original 1e-5)?
- **Setup.** lr 1e-5, 2,000 steps, layer 12, seed 1. Run `20260826_044036`.
- **Result.** Neither peaks at +0.094 (step 1,200) against +0.085 at 5e-5
  (step 200). All pairs never goes negative within 2,000 steps (+0.018 at the
  end) where 5e-5 crosses zero by step 600.
- **Conclusion.** Not an artefact; the learning rate sets the timescale. The
  all-pairs collapse at 5e-5 is not the same phenomenon as the original
  pipeline's.
- findings: "The original learning rate makes our result stronger".

### E08. Long run at 5e-5

- **Question.** What happens if training simply continues?
- **Setup.** 4,000 steps, lr 5e-5, layer 12, seed 1, evaluated every 200.
  Run `20260826_022741`.
- **Result.** Neither +0.085 at step 200, decaying to a plateau near +0.025
  from step 2,000. All pairs crosses zero near step 600 and settles at -0.017.
  Control flat throughout.
- **Conclusion.** Most of the early gain is transient at this learning rate;
  about a third persists. Plateau value uncertain at one seed (E10).
- findings: "The effect decays at our learning rate but not at the original one".

### E09. Original schedule on this pipeline

- **Question.** Do training length and learning rate explain why real-sentence
  RSR fails in the original pipeline?
- **Setup.** lr 1e-5, 8,000 steps, layer 12, seed 1, evaluated every 400.
  Run `20260826_055015`.
- **Result.** Neither peak +0.095 (step 1,200), +0.067 at step 8,000, mean
  +0.072 over the last 2,000 steps. All pairs -0.002 at the end. Both -0.062.
- **Conclusion.** Neither length nor learning rate explains the disagreement.
  At long training the result is a dissociation: supervised pairs worse, all
  pairs at baseline, unsupervised pairs retain the gain.
- **Later.** Extended to 3 seeds: Neither +0.065 at step 8,000.
- **Note (E20).** The Both decline is now attributable at least in part to
  verbs never being trained.
- findings: "The original configuration, run on our code".

### E10. Same-seed reproducibility

- **Question.** How far can a single run be trusted?
- **Setup.** The 2,000-step and 8,000-step runs at lr 1e-5 share every input
  up to step 2,000; compare trajectories.
- **Result.** All-pairs divergence 0.0002 at step 400, 0.003 at 800, 0.009 at
  1,600, 0.015 at 2,000. Matches the 0.003 floor from duplicate controls.
- **Conclusion.** GPU nondeterminism amplifies with training. Quantitative
  claims past step 1,000 need several seeds.
- findings: "How far can a single run be trusted?"

### E11. MEN transfer

- **Question.** Is the effect specific to SimLex?
- **Setup.** Evaluation only, strict-pool checkpoint, seed 1, MEN partitioned
  by the same supervision vocabulary.
- **Result.** Neither +0.061 (1,569 pairs), All +0.063, Both +0.027. Vanilla
  and control identical on All (0.7305).
- **Conclusion.** Transfers to a relatedness benchmark never used for
  training or selection, with the same ordering (Both gains least). The
  expected failure on relatedness did not occur: the defensible claim is a
  general improvement in lexical representation.
- findings: "The effect is not specific to SimLex".

### E12. Templates against real sentences (BERT)

- **Question.** The two original RSR arms differ only in training context
  (template or sentence), and the original evaluates bare words by default. Is
  the original real-sentence collapse a measurement artefact?
- **Setup.** Both training regimes, 3 seeds, 2,000 steps, every checkpoint
  scored in sentence, bare and template contexts.
- **Result.** Neither, sentence evaluation: sentence-trained +0.045, template
  +0.019 (p = 0.037). Bare evaluation: +0.129 against +0.116 (p = 0.29), with
  seed ranges of 0.13. Template-trained RSR reaches lower loss (0.150 against
  0.191). Continued MLM alone halves the control's template probe (0.503 to
  0.250).
- **Conclusion.** Bare evaluation cannot separate the regimes and is too
  unstable for effect sizes. Template training fits better and generalises
  worse. The original real-sentence collapse is not explained by evaluation
  context: real-sentence RSR is healthy in all three.
- findings: "Templates vs real sentences".

### E13. GPT-2 layer probe

- **Question.** Which GPT-2 layer is a usable lexical read-out?
- **Setup.** All 999 pairs, 20 sentences per word, every layer of untrained
  GPT-2. Run `results/gpt2_layer_probe_20260908_005631.csv`.
- **Result.** All pairs 0.529 at layer 6, 0.161 at layer 12, where mean cosine
  is 0.994 +/- 0.006. BERT's layer 12 reads 0.480.
- **Conclusion.** GPT-2's final layer is collapsed (Ethayarajh, 2019); layer 6
  is the functional counterpart of BERT's layer 12 and is used for the main
  GPT-2 runs.
- findings: "GPT-2: the last layer is not a usable read-out".

### E14. GPT-2 main result

- **Setup.** Same code, causal LM anchor, layer 6, 3 seeds, 2,000 steps.
  Runs `gpt2_continued_*_main_s{1,2,3}`.
- **Result.** Early-stopped: Neither +0.026 (range 0.004, p = 0.003), Both
  -0.052, All -0.010. Endpoint: Neither +0.021, Both -0.088. MEN Neither
  +0.051.
- **Conclusion.** The effect transfers to a causal model at about a third of
  BERT's SimLex size, near BERT's size on MEN. The Neither/Both gap matches
  BERT, reached by larger losses on Both.
- findings: "GPT-2: the result transfers".

### E15. Retrofitting and post-specialisation baselines

- **Question.** Does a standard specialisation method reach unsupervised
  words?
- **Setup.** Retrofitting over a k-NN graph; post-specialisation as a residual
  two-layer map from original to retrofitted vectors, fitted on supervised
  words and applied to all.
- **Result.** Change against stock BERT on Neither: retrofitting +0.000
  (forced), post-specialisation -0.026, RSR +0.091. Post-specialisation
  recovers All from -0.223 to -0.046.
- **Conclusion.** Only RSR helps the unsupervised words. Narrow scope:
  post-specialisation of this retrofitting stage, not a reimplementation of
  Vulic et al. (2018).
- findings: "Post-specialisation reaches the unsupervised words".

### E16. GPT-2 long run

- **Setup.** 8,000 steps at 1e-5, layer 6, 3 seeds, endpoint.
- **Result.** Neither +0.023 at the end (p = 0.006), Both -0.098 and still
  falling, All -0.026. MEN Neither +0.060.
- **Conclusion.** Same three-way pattern as BERT; MEN transfer does not erode.
- findings: "GPT-2 extended training".

### E17. GPT-2 templates against real sentences

- **Result.** Neither, sentence evaluation: +0.021 against +0.016 (p = 0.37).
  Bare evaluation of untouched GPT-2: -0.045.
- **Conclusion.** "Template training is worse" is BERT-specific. Bare scoring
  on a causal model is a null instrument.
- findings: "GPT-2 templates vs real sentences".

### E18. GPT-2 layer 12

- **Question.** On BERT a worse starting layer gained more. Does GPT-2's
  collapsed layer 12 gain more than layer 6?
- **Result.** Neither +0.123 early-stopped, +0.136 endpoint; every partition
  rises (Both +0.060, One +0.217).
- **Conclusion.** The prediction holds, but the mechanism is recovery from
  anisotropy, not selective reorganisation. Reported separately.
- findings: "GPT-2 layer 12".

### E19. Supervision audit

- **Date.** 8 November 2026. No training.
- **Question.** What does the supervision actually train on?
- **Setup.** `audit_supervision.py`: supervision composition, SimLex overlap,
  partition by part of speech, 1,000 simulated batches per configuration.
- **Result.**
  - 274 of 297 Both pairs were scored training pairs (170 SimVerb, 104
    THINGS); 23 Both pairs unseen. One and Neither clean by construction.
    Excluding them removes no word.
  - SimVerb is 0.2% of scored pairs per batch (table in 1.4).
  - Both is 65% verbs; Neither is nouns and adjectives.
  - Pair-first sampling alone makes SimVerb's share smaller (0.03 pairs per
    batch); only a THINGS sample gives it a real share.
  - The local copy of the original pipeline (1 July) had no SimLex filter; the
    upstream repository added one on 4 July (lemma-based pair matching after
    augmentation).
- **Conclusion.** Verbs were supervised in name only, and the Both partition
  overlapped training. Switches added for each issue (section 1.3, 1.4, 3).
- findings: "Supervision balance".

### E20. Supervision and sampling arms

- **Date.** 8 to 10 November 2026. Runner
  `experiments/runners/run_supervision_checks.sh`.
- **Question.** Does excluding SimLex pairs change the result? Does giving
  SimVerb a real share of the signal change Both, and what does it do to
  Neither? Which component is responsible?
- **Setup.** BERT, 2,000 steps, endpoint, lr 5e-5, layer 12, sentence
  context. Arms:
  - endpoint: reported configuration.
  - clean: `--exclude-simlex-pairs --split-sentences`.
  - pairs: clean + `--sampler pairs --things-pairs 10000`.
  - inflect: pairs + `--inflect-verbs` (scored against the pairs control).
  - pairs_full: clean + `--sampler pairs` (scored against the clean control).
  - share33: pairs_full + `--simverb-share 0.33` (against clean).
- **Result.** 5 seeds for the first four, 1 seed for the last two.

  | arm | All | Both | One | Neither | Both V | Both N | Neither N | Neither A |
  |---|---|---|---|---|---|---|---|---|
  | endpoint | -0.016 | -0.066 | -0.044 | +0.045 | +0.104 | -0.021 | +0.028 | +0.082 |
  | clean | -0.019 | -0.053 | -0.039 | +0.039 | +0.134 | -0.101 | +0.028 | +0.068 |
  | pairs | +0.123 | +0.212 | +0.019 | +0.155 | +0.491 | -0.075 | +0.128 | +0.202 |
  | inflect | +0.144 | +0.219 | +0.025 | +0.188 | +0.502 | -0.065 | +0.157 | +0.224 |
  | pairs_full (s1) | -0.020 | -0.047 | -0.050 | +0.036 | +0.123 | -0.081 | +0.023 | +0.075 |
  | share33 (s1) | +0.091 | +0.176 | -0.032 | +0.142 | +0.502 | -0.081 | +0.108 | +0.201 |

  One-sample p for Neither: endpoint 0.0016, clean 0.0047, pairs 3e-5,
  inflect 2e-6. Paired, 5 seeds: endpoint to clean, Neither -0.006
  (p = 0.49); clean to pairs, Neither +0.116 (p = 1e-6), Both +0.265
  (p = 1e-7), All +0.141 (p = 8e-6); pairs to inflect, Neither +0.033
  (p = 0.011), All +0.021 (p = 0.007), Both +0.008 (p = 0.23).

  Absolute Neither (mean): vanilla 0.513, control 0.526, RSR 0.564 (clean),
  0.681 (pairs), 0.714 (inflect). Neither peaks at step 280 to 340 in the old
  sampling and step 1,060 to 1,080 in the pairs arms; the end-of-training drop
  from the peak is 0.026 to 0.037 throughout. Final anchor loss within 0.05 of
  the control in every arm.
- **Conclusions.**
  1. The SimLex overlap and the shared sentence pool had no measurable effect
     on Neither.
  2. Without early stopping the reported-configuration Neither gain is +0.045,
     against +0.070 early-stopped.
  3. Putting SimVerb pairs in the batches raises Neither about fourfold and
     turns Both and All strongly positive. The Both gain is a verb gain
     (+0.49); Both nouns stay negative.
  4. Pair-first sampling over all of THINGS does nothing (0.03 SimVerb pairs
     per batch). A third of draws from SimVerb, with all of THINGS kept,
     recovers most of the gain at 3.4% of the scored pairs. The effect follows
     the presence of SimVerb pairs, not their share of the loss and not the
     reduction of THINGS. (One seed each; to be repeated.)
  5. Neither adjectives, never supervised in any form, gain +0.20. Working
     explanation: SimVerb was rated under SimLex's similarity instructions,
     THINGS is closer to category structure, and RSR on SimVerb teaches the
     similarity/relatedness distinction itself.
- **Not done.** MEN, brain and GLUE on these checkpoints; GPT-2; more seeds of
  pairs_full and share33.
- findings: "Supervision balance".

### E21. MEN under the recommended configuration

- **Date.** 24 November 2026. Runner
  `experiments/runners/run_main_config_followups.sh`, steps `men_main` and
  `men_endpoint`.
- **Question.** Does the larger SimLex gain of the recommended configuration
  carry over to a relatedness benchmark, as the reported sampling's did (E11)?
- **Setup.** Evaluation only. E20 inflect checkpoints against the pairs
  control, seeds 1 to 5; E20 endpoint checkpoints against their own control
  for comparison. `eval_men_transfer.py` now rebuilds each run's supervision
  from its saved config, so the partition follows that run's vocabulary and
  the sentence split. Main configuration: 5,111 supervised words, MEN
  partitions 379 / 1,213 / 1,408 (Both / One / Neither). Reported sampling:
  2,652 words, 294 / 1,122 / 1,584.
- **Result.** RSR minus control, mean over 5 seeds, one-sample p:

  | configuration | All | Both | One | Neither |
  |---|---|---|---|---|
  | recommended | +0.040 (p = 7e-4) | +0.019 (p = 0.12) | +0.055 (p = 7e-4) | +0.032 (p = 0.002) |
  | reported sampling | +0.071 (p = 3e-4) | +0.047 (p = 3e-4) | +0.083 (p = 2e-4) | +0.063 (p = 8e-4) |

  Absolute All: vanilla 0.712, control 0.721, RSR 0.761 (recommended);
  vanilla 0.712, control 0.708, RSR 0.779 (reported). Neither positive in
  every seed in both configurations.
- **Conclusions.**
  1. The transfer to MEN survives the change of supervision, on every
     partition and every seed.
  2. It is about half as large. SimLex Neither goes from +0.045 to +0.188
     while MEN Neither goes from +0.063 to +0.032, so the two benchmarks move
     in opposite directions.
  3. This fits the working explanation in E20: SimVerb pairs teach the
     similarity versus relatedness distinction, which helps a similarity
     benchmark and costs some relatedness. THINGS, which dominates the
     reported sampling, is closer to relatedness.
- **Results.** `results/men_transfer_main.csv`,
  `results/men_transfer_endpoint.csv`.
- **Later note.** Conclusion 3 is withdrawn by E27: MEN holds as SimVerb
  rises when all of THINGS is kept.

### E22. Stem check under the recommended configuration

- **Question.** With inflected verb forms supervised, could Neither gain
  through morphological neighbours (a Neither pair whose word shares a stem
  with a supervised form)?
- **Setup.** `eval_strict_neither.py`, inflect against pairs, seeds 1 to 5.
  Drops every Neither pair with a word that shares a stem with any
  supervised word, leaving 446 of 481.
- **Result.** Neither +0.188 on all 481 pairs, +0.186 on the 446 strict pairs
  (difference -0.002). Per seed strict: +0.174, +0.197, +0.195, +0.180,
  +0.183.
- **Conclusion.** Not stem leakage.

### E23. Brain RSA under the recommended configuration

- **Question.** Does the unsupervised gain show up in Pereira brain data under
  the recommended configuration?
- **Setup.** `eval_pereira_bert.py --layers all --sup-config`, seeds 1 to 3,
  vanilla, pairs control and inflect RSR. The partition follows the run's
  vocabulary: 57 of 180 Pereira concepts supervised (49 under the reported
  sampling). Per-subject RSA averaged over the three training seeds, then a
  paired t-test over the 10 subjects.
- **Result.** RSR minus control:

  | layer | All | Both | One | Neither |
  |---|---|---|---|---|
  | 7 | +0.028 (p = 0.030) | +0.001 | +0.006 | **+0.032 (p = 0.002, 9/10)** |
  | 9 | +0.019 (p = 0.057) | +0.005 | +0.001 | +0.018 (p = 0.038, 8/10) |
  | 12 | +0.026 (p = 0.044) | -0.004 | +0.008 | **+0.027 (p = 0.011, 9/10)** |

  Neither at layer 12 against vanilla: +0.023 (p = 0.035). Single seeds at
  layer 12: Neither +0.032 (p = 6e-4), +0.016 (p = 0.17), +0.033 (p = 0.029).
  At layer 12 the control falls below vanilla on All (0.016 against 0.030);
  at layer 7 it does not.
- **Conclusions.**
  1. The brain result holds: the gain is on unsupervised concepts, Both and
     One are flat.
  2. It is smaller than E04's single-seed +0.054. Part of that is seed
     variance (one of three seeds misses significance on its own); part may
     be the same similarity for relatedness trade as E21, since RSA with
     sentence-level betas is not a similarity judgement.
  3. The effect is clearest at layer 7, where the control is not degraded, so
     it is not only a repair of the layer-12 read-out.
- **Results.** `brain_and_glue/results/brain_main_s{1,2,3}/`,
  `results/brain_main_seedmean.csv`.

### E24. Seeds 2 and 3 of the SimVerb-presence arms

- **Date.** 24 to 25 November 2026. Step `split_arms_s23` of the follow-up
  runner.
- **Question.** E20 conclusion 4 (the gain follows the presence of SimVerb
  pairs, not their share and not the THINGS cut) rested on one seed per arm.
  Does it hold over three?
- **Setup.** pairs_full and share33 as in E20, seeds 2 and 3, scored against
  the clean control of the same seed.
- **Result.** RSR minus clean control, final step, 3 seeds:

  | arm | All | Both | One | Neither | Both V | Both N | Neither N | Neither A |
  |---|---|---|---|---|---|---|---|---|
  | clean (s1 to 3) | -0.020 | -0.048 | -0.049 | +0.039 | +0.138 | -0.094 | +0.019 | +0.084 |
  | pairs_full | -0.026 | -0.065 | -0.043 | +0.031 | +0.095 | -0.079 | +0.019 | +0.064 |
  | share33 | **+0.086** | **+0.176** | -0.024 | **+0.131** | **+0.513** | -0.042 | +0.095 | +0.198 |

  Neither per seed: pairs_full +0.036, +0.036, +0.020; share33 +0.142,
  +0.129, +0.122. SimVerb pairs per batch: pairs_full 0.01 to 0.03, share33
  4.6 to 5.0. Paired against the clean RSR arm over seeds 1 to 3: pairs_full
  Neither p = 0.45, All p = 0.54; share33 Neither p = 0.016, All p = 0.010.
- **Conclusion.** Confirmed. Pair-first sampling with all of THINGS is
  indistinguishable from word sampling; a third of draws from SimVerb, with
  all of THINGS kept, recovers about 70% of the Neither gain of the
  recommended configuration and all of the Both verb gain. The THINGS cut is
  not needed; the SimVerb pairs are.
- **Results.** `results/bert_continued_rsr_sup_{pairs_full,share33}_s{1,2,3}_*.csv`.
- **Later note.** "Not their share" is revised by E27: with shares 0.10 and
  0.66 added, the gain grows with the number of SimVerb pairs per batch.

### E25. GPT-2 under the recommended configuration

- **Date.** 25 November 2026. Steps `gpt2_main_s{1,2,3}`.
- **Question.** Does the effect of putting SimVerb pairs in the batches carry
  over to a causal model? E14 found GPT-2 at about a third of BERT's SimLex
  gain under word sampling.
- **Setup.** `gpt2_rsr_continued_pretraining.py --regime compare --lambda 0.7
  --steps 2000 --rsr-layer 6` with the recommended switches, seeds 1 to 3,
  final step. 23,310 scored pairs over 5,111 words, the same supervision as
  the BERT runs. MEN partitioned by the same vocabulary inside the script.
- **Result.** RSR minus control, 3 seeds:

  | | All | Both | One | Neither | Both V | Both N | Neither N | Neither A |
  |---|---|---|---|---|---|---|---|---|
  | SimLex | +0.064 | +0.117 | +0.032 | **+0.078** | +0.382 | -0.165 | +0.055 | +0.110 |
  | MEN | +0.053 | +0.042 | +0.063 | +0.050 | | | | |

  SimLex Neither per seed +0.083, +0.066, +0.086 (p = 0.006); All p = 0.003;
  Both p = 0.002. MEN Neither +0.057, +0.036, +0.057 (p = 0.019). Absolute
  SimLex Neither: control 0.532, RSR 0.610. Peak Neither gain +0.086 to
  +0.098, at steps 900 to 2,000. Final CLM loss 0.25 to 0.33 above the
  control (0.10 to 0.13 in E14; the BERT MLM cost in E20 is 0.03 to 0.08).
- **Conclusions.**
  1. The supervision effect transfers to GPT-2. SimLex Neither goes from
     +0.021 (E14, endpoint) to +0.078, nearly fourfold, as in BERT; Both flips
     from -0.088 to +0.117, again through the verbs, and supervised nouns get
     worse.
  2. GPT-2 stays at about 40% of BERT's SimLex gain.
  3. GPT-2's MEN gain does not fall (+0.051 in E14, +0.050 here). The
     similarity versus relatedness trade of E21 is a BERT result, not a
     general consequence of the supervision.
  4. The anchor pays more than in BERT. Downstream checks for GPT-2 have not
     been run.
- **Results.** `results/gpt2_continued_{rsr,baseline}_sup_main_s{1,2,3}_*.csv`,
  logs `logs/followups/gpt2_main_s*.log`.

### E26. GLUE under the recommended configuration

- **Date.** 25 November 2026. Steps `glue_main_s{1,2,3}`. The first attempt
  failed at import (transformers loading TensorFlow under Keras 3); the
  runner now sets `USE_TF=0`.
- **Question.** Does the recommended configuration cost downstream ability,
  as the reported sampling did not (E04)?
- **Setup.** `eval_downstream_bert.py`, BERT inflect RSR and pairs control
  from seeds 1 to 3, vanilla once; MRPC, RTE, CoLA, STS-B, SST-2; 3
  fine-tuning seeds per checkpoint, 10% warmup, degenerate-run detector.
- **Result.** Mean over 3 checkpoints x 3 fine-tuning seeds; paired over
  checkpoints:

  | task | vanilla | control | RSR | RSR - control | per checkpoint | p |
  |---|---|---|---|---|---|---|
  | MRPC (acc) | 0.862 | 0.840 | 0.818 | **-0.022** | -0.020, -0.025, -0.021 | 0.003 |
  | RTE | 0.668 | 0.629 | 0.640 | +0.012 | +0.004, +0.020, +0.011 | 0.14 |
  | CoLA | 0.575 | 0.562 | 0.572 | +0.010 | -0.001, +0.018, +0.013 | 0.23 |
  | STS-B | 0.892 | 0.889 | 0.889 | +0.001 | | 0.87 |
  | SST-2 | 0.928 | 0.924 | 0.924 | +0.000 | | 1.00 |

  MRPC F1: control 0.890, RSR 0.875. 0 degenerate runs of 105.
- **Conclusions.**
  1. Parity on four of five tasks; CoLA, the syntactic task, is not hurt.
  2. MRPC loses 0.022 accuracy in every checkpoint. Under the reported
     sampling it was +0.002 (E04). The stronger verb supervision has a small,
     consistent cost on paraphrase detection. This is the one measured price
     of the recommended configuration.
  3. Continued MLM training alone already costs MRPC 0.022 and RTE 0.039
     relative to vanilla, so both RSR and control sit below stock BERT there.
- **Results.** `brain_and_glue/results/downstream_main_s{1,2,3}/downstream_results.json`.

### E27. SimVerb share with all of THINGS

- **Date.** 26 November 2026. Runner `experiments/runners/run_simverb_share.sh`,
  steps `men_split`, `share{10,66}_s{1,2,3}`, `men_sweep`.
- **Question.** E21 and E26 found two costs of the recommended configuration
  (MEN halves, MRPC -0.022). share33 keeps all of THINGS. How does the gain
  depend on the amount of SimVerb in the batches, and do the costs follow
  SimVerb or the THINGS cut?
- **Setup.** Pair-first sampling over all THINGS pairs, SimLex pairs
  excluded, split sentences, `--simverb-share` 0.10 and 0.66, seeds 1 to 3,
  RSR only, scored against the clean control of the same seed. With
  pairs_full (share 0) and share33 (E24) this gives four points. MEN on all
  four arms against the clean control, partitions from the 2,652-word
  vocabulary (294 / 1,117 / 1,569).
- **Result, SimLex.** RSR minus clean control, final step, 3 seeds; p paired
  against the clean RSR arm:

  | arm | SimVerb pairs per batch | THINGS pairs per batch | All | Both | One | Neither | Both V | Both N | Neither N | Neither A |
  |---|---|---|---|---|---|---|---|---|---|---|
  | pairs_full | 0.03 | 275 | -0.026 | -0.065 | -0.043 | +0.031 | +0.095 | -0.079 | +0.019 | +0.064 |
  | share10 | 1.3 | 224 | -0.003 | -0.005 | -0.046 | +0.048 | +0.223 | -0.081 | +0.028 | +0.092 |
  | share33 | 4.6 | 128 | +0.086 | +0.176 | -0.024 | +0.131 | +0.513 | -0.042 | +0.095 | +0.198 |
  | share66 | 10.5 | 36 | **+0.110** | **+0.195** | -0.003 | **+0.160** | **+0.518** | -0.060 | +0.127 | +0.223 |

  Neither per seed: share10 +0.046, +0.054, +0.043; share66 +0.163, +0.167,
  +0.152. Against clean RSR: share10 Neither p = 0.45, Both V p < 0.001;
  share66 Neither p = 0.004, All p = 0.015. share66 against share33: Neither
  +0.029 (p = 0.027), Both V +0.005 (p = 0.73). MLM loss within 0.04 of the
  control in every arm.
- **Result, MEN.** RSR minus clean control, 3 seeds, one-sample p:

  | arm | All | Both | One | Neither |
  |---|---|---|---|---|
  | pairs_full | +0.061 | +0.039 | +0.078 | +0.051 (p = 0.004) |
  | share10 | +0.066 | +0.047 | +0.085 | +0.055 (p = 0.009) |
  | share33 | +0.066 | +0.042 | +0.077 | +0.064 (p < 0.001) |
  | share66 | +0.072 | +0.034 | +0.091 | +0.065 (p = 0.003) |
  | recommended (E21, for reference) | +0.040 | +0.019 | +0.055 | +0.032 |

- **Conclusions.**
  1. The gain follows the number of SimVerb pairs per batch, not just their
     presence. About one per batch moves the supervised verbs (Both V +0.22)
     but not the unsupervised words; most of the Neither gain arrives by
     about five per batch; doubling again adds +0.03. This revises E20
     conclusion 4 and E24.
  2. With all of THINGS kept, share66 reaches Neither +0.160 against +0.188
     for the recommended configuration, without the THINGS cut or the
     inflected forms.
  3. MEN does not fall as SimVerb rises. It rises slightly, and share66, with
     more SimVerb pairs per batch than the recommended configuration (10.5
     against 7.4), keeps the MEN transfer of the reported sampling. The MEN
     halving in E21 therefore comes from the 10k THINGS sample or the
     inflected forms, not from SimVerb. E21 conclusion 3 is withdrawn. E29
     separates the two.
- **Results.** `results/bert_continued_rsr_sup_share{10,66}_s{1,2,3}_*.csv`,
  `results/men_transfer_split.csv`, `results/men_transfer_sweep.csv`.

### E28. GLUE on share33

- **Date.** 26 November 2026. Steps `glue_share33_s{1,2,3}`.
- **Question.** Does keeping all of THINGS avoid the MRPC cost of the
  recommended configuration (E26)?
- **Setup.** As E26, share33 RSR against the clean control, seeds 1 to 3, 3
  fine-tuning seeds per checkpoint. Vanilla from E26.
- **Result.** Mean over 3 checkpoints x 3 fine-tuning seeds; paired over
  checkpoints:

  | task | vanilla | control | RSR | RSR - control | per checkpoint | p | recommended (E26) |
  |---|---|---|---|---|---|---|---|
  | MRPC (acc) | 0.862 | 0.842 | 0.835 | -0.007 | -0.003, +0.000, -0.017 | 0.33 | -0.022 |
  | RTE | 0.668 | 0.639 | 0.670 | +0.032 | +0.024, +0.048, +0.023 | 0.06 | +0.012 |
  | CoLA | 0.575 | 0.562 | 0.571 | +0.009 | -0.009, +0.016, +0.019 | 0.44 | +0.010 |
  | STS-B | 0.892 | 0.889 | 0.889 | -0.000 | | 0.99 | +0.001 |
  | SST-2 | 0.928 | 0.924 | 0.926 | +0.002 | | 0.14 | +0.000 |

  MRPC F1: control 0.890, RSR 0.886. 0 degenerate runs of 90.
- **Conclusions.**
  1. The MRPC cost is a third of its size and no longer consistent (one
     checkpoint of three). share33 is at parity on every task.
  2. Together with E27, both measured costs of the recommended configuration
     track the THINGS cut (or the inflected forms), not the SimVerb pairs.
  3. share66 was not run on GLUE. It is the arm to test next, since it is the
     closest to the recommended configuration on SimLex.
- **Results.** `brain_and_glue/results/downstream_share33_s{1,2,3}/downstream_results.json`.

### E29. MEN on the pairs arm

- **Date.** 26 November 2026. Step `men_pairs`.
- **Question.** The recommended configuration differs from share33 and
  share66 in two ways: the 10k THINGS sample and the inflected SimVerb forms.
  Which one costs MEN?
- **Setup.** E20 pairs checkpoints (10k THINGS, no inflection) against the
  pairs control, seeds 1 to 5. Vocabulary 2,651 words, partitions 294 /
  1,117 / 1,569. The pairs and clean controls are the same model (the control
  does not use the RSR sampler); their MEN scores agree to four decimals.
- **Result.** RSR minus control, 5 seeds, one-sample p:

  | arm | All | Both | One | Neither |
  |---|---|---|---|---|
  | pairs (10k THINGS) | +0.065 (p = 1e-4) | +0.020 (p = 0.035) | +0.080 (p = 7e-5) | +0.063 (p = 9e-5) |
  | inflect (E21) | +0.040 | +0.019 | +0.055 | +0.032 |

  Neither per seed: +0.069, +0.074, +0.059, +0.054, +0.057. All is the same
  2,980 pairs in both rows and the control is shared, so the 0.025 drop on
  All is a direct comparison of the two RSR arms.
- **Conclusions.**
  1. The 10k THINGS sample does not cost MEN. The inflected SimVerb forms do:
     they add 0.033 on SimLex Neither (E20) and take 0.025 off MEN All and
     about 0.03 off MEN Neither.
  2. Part of the Neither drop in E21 may be the partition: inflection doubles
     the vocabulary, so the Neither set changes. The All row has no such
     caveat.
  3. Whether the MRPC cost (E26) also comes from the inflections is not yet
     known; GLUE on the pairs arm would settle it.
- **Results.** `results/men_transfer_pairs.csv`.

---

## 5. Corrections and retracted claims

| claim | status | evidence |
|---|---|---|
| "No SimLex pair is ever trained on" | wrong for Both before Nov 2026 (274 of 297); true with `--exclude-simlex-pairs` | E19, E20 |
| The loss is a soft Spearman correlation | as run, effectively Pearson (sigmoid fallback at strength 1) | 1.1 |
| Headline +0.070 on Neither | early-stopped on the test set; endpoint +0.045 (5 seeds) | E20 |
| Verbs (SimVerb) are part of the supervision | only nominally: 0.26 scored pairs per batch | E19 |
| GPT-2 layer 12 is negatively correlated with SimLex (-0.103) | artefact of a 150-pair CPU probe; full probe gives +0.161 | E13 |
| "Layer weighting does not help" | too broad; true of every read-out tested, and only averaging is a clear negative | E05 |
| The all-pairs collapse at 5e-5 reproduces the original | not the same phenomenon; absent at 1e-5 | E07 |
| "Template training produces a worse model" | BERT only | E12, E17 |
| The original real-sentence collapse is an evaluation artefact | rejected | E12 |
| The gain follows the presence of SimVerb pairs, not their share | revised: it grows with SimVerb pairs per batch, little below one, most by about five | E24, E27 |
| More SimVerb trades relatedness (MEN) for similarity | withdrawn: MEN holds at 10.5 SimVerb pairs per batch with all of THINGS, and with the 10k THINGS sample; the MEN cost comes from the inflected forms | E21, E27, E29 |

---

## 6. Open questions and planned work

Highest priority first.

1. **Choose the configuration to report.** The inflected forms buy +0.033
   on SimLex Neither and cost about 0.03 on MEN (E29). Candidates without
   them: pairs (10k THINGS, Neither +0.155) and share66 (all THINGS,
   +0.160), both with the full MEN transfer. GLUE on pairs (already trained)
   locates the MRPC cost; GLUE on share66 completes its profile. About 3.5
   hours each.
2. **Why the inflected forms cost MEN.** They double the vocabulary with
   forms rated by copying the base pair's score; check whether the cost
   follows the number of inflected pairs per batch.
3. **GPT-2 anchor cost.** CLM loss rises 0.29 above the control under the
   recommended configuration (E25). Check perplexity on held-out text, or
   a downstream task, before claiming no damage for GPT-2.
4. **Soft rank.** Install torchsort or raise the sigmoid strength, and rerun
   one seed, so the loss is a rank correlation as described, or describe it as
   Pearson.
5. **Checkpoint rule.** Fix the reporting step in advance or select on a dev
   set that is not SimLex (WordSim-353 is available). Both depends on length.
6. **Nouns.** THINGS-supervised nouns fall on SimLex in every arm. Test a
   THINGS-only run with held-out THINGS words, scored on held-out pairs and on
   Pereira (mostly concrete nouns).
7. **Identical anchor streams** for RSR and control (separate seeded
   generator), before the next full set of runs.
8. **Statistics.** Bootstrap intervals; state Neither as the pre-specified
   test.
9. **Backfill.** The lambda and combination sweep that selected weighted 0.7.
10. **fMRI teacher.** Region, noise ceiling, control design, sampler.
11. **Original-pipeline disagreement.** Remaining untested differences:
    RoBERTa against BERT, layer 5 against 12, alternating against weighted
    losses. E20 suggests supervision composition may account for much of it.

---

## 7. Reproduction

```bash
pip install -r requirements.txt

# what a configuration trains on, no model loaded
python experiments/rsr_pretraining/audit_supervision.py \
    --exclude-simlex-pairs --sampler pairs --things-pairs 10000 --inflect-verbs

# recommended configuration, one seed, RSR and matched control
python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py \
    --regime compare --lambda 0.7 --seed 1 \
    --sentences-dir artifacts/sentences_strict \
    --exclude-simlex-pairs --split-sentences \
    --sampler pairs --things-pairs 10000 --inflect-verbs

# all E20 arms, resumable
SEEDS="1 2 3" MORE_SEEDS="4 5" bash experiments/runners/run_supervision_checks.sh
```

Inflected forms need sentences first (README, Switches). Unit tests:
`python -m unittest discover -s tests -t .`
