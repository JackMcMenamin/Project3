# Representational Similarity Regularisation

Experiment 3 of the PhD. RSR adds a second training objective to a pre-trained
language model: make its word-pair similarities rank the way human similarity
judgements do. The similarity loss is a differentiable Spearman correlation over
the cosine similarities of the model's contextual word vectors, supervised by
human ratings for about 2,600 words and combined with the model's own
pre-training objective.

The question the project answers is whether that reorganises the model's lexical
space beyond the words it was given ratings for. It does, and the effect is
concentrated there. On held-out SimLex-999, split by whether each pair's words
were ever supervised, the improvement lands almost entirely on pairs where
neither word was (+0.070 over a matched control, five seeds), while the
supervised pairs do not move. The same signature appears on MEN, in
representational similarity analysis against fMRI data, and in GPT-2 with a
causal-LM anchor in place of MLM. GLUE is unaffected. Post-processing
alternatives (retrofitting, post-specialisation) cannot reach the unsupervised
words at all.

## Layout

```
experiments/rsr_pretraining/   the method: BERT and GPT-2 continued pre-training
                               with RSR, plus evaluations, baselines and plots
experiments/runners/           multi-seed batch scripts
experiments/contextual/        harvesting the sentence pool from Wikipedia
experiments/word2vec/          the original word-level pilot
brain_and_glue/                RoBERTa replication, brain RSA, GLUE
notes/findings.md              research notes, chronological, with the numbers
src/rsr/                       shared library for the earlier experiments
docs/  figures/  archive/      reports, earlier figures, superseded code
```

## Where to start

- `notes/findings.md` for what was found, in what order, and what was later
  corrected.
- `experiments/rsr_pretraining/bert_rsr_continued_pretraining.py` for the
  method itself. `Config` and `rsr_loss_for_batch` are the parts that matter.
- `experiments/rsr_pretraining/gpt2_rsr_continued_pretraining.py` for what had
  to change for a causal model, explained in its header.

## Running

```bash
# BERT: matched MLM control + RSR, one seed
python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py \
    --regime compare --combine weighted --lambda 0.7 --steps 800 \
    --sentences-dir artifacts/sentences_strict --seed 1

# GPT-2: same recipe, read out at layer 6 (its final layer is unusable for this)
python experiments/rsr_pretraining/gpt2_rsr_continued_pretraining.py \
    --regime compare --lambda 0.7 --rsr-layer 6 \
    --sentences-dir artifacts/sentences_strict --seed 1

# baselines and transfer, evaluation only
python experiments/rsr_pretraining/eval_retrofit_baseline.py
python experiments/rsr_pretraining/eval_postspec_baseline.py
python experiments/rsr_pretraining/eval_men_transfer.py
```

Multi-seed runs go through `experiments/runners/` and are resumable: each arm
skips its own finished output, so an interrupted batch picks up where it stopped.

Pass the sentence pool explicitly. All reported numbers use the strict
full-Wikipedia pool at `artifacts/sentences_strict`, built once with
`experiments/contextual/harvest_sentences_strict.py` and
`filter_sentence_pool.py`. Corpora, harvested sentences, checkpoints and per-run
CSVs are not versioned.

```bash
pip install torch transformers numpy pandas scipy scikit-learn tqdm matplotlib torchsort
```

## Protocol

Every reported number is a mean over seeds against a control that differs from
the RSR arm in exactly one thing, the RSR term. Five seeds for the SimLex
headline, three for everything else. Early stopping selects the best all-pairs
checkpoint; the notes report what happens without it.
