# Representational Similarity Regularisation

RSR adds a second objective to a pre-trained language model: its word-pair
similarities should rank the way human similarity ratings do. The RSR term is a
differentiable rank correlation between the cosines of the model's contextual
word vectors and the ratings, for about 2,650 supervised words (SimVerb-3500
and THINGS). It is combined with the model's own pre-training loss, and the
whole model trains.

The question is whether this reorganises the lexical space beyond the words it
was given ratings for. On SimLex-999, split by how many of each pair's words
were supervised, the gain over a matched control lands on the pairs where
neither word was supervised (BERT +0.070, five seeds; GPT-2 +0.026, three
seeds), while the supervised pairs do not improve. MEN, held out entirely,
shows the same pattern. With the supervision rebalanced so verb pairs are
actually trained (see Switches), the BERT gain on unsupervised pairs rises to
+0.155 to +0.188 at the final step, five seeds.

`notes/technical_report.md` is the structured record: method, current
numbers, every experiment with its motivation and result, and corrections.
`notes/findings.md` is the working notebook and `notes/todo.md` the open
items.

## Layout

```
experiments/rsr_pretraining/    BERT and GPT-2 continued pre-training with RSR,
                                evaluations, baselines, summaries and plots
  rsr_core/                     shared by both: data, partition, loss, tables
experiments/runners/            multi-seed batch scripts
experiments/contextual/         builds the Wikipedia sentence pool
experiments/word2vec/           the word-level Word2Vec pilot
brain_and_glue/                 brain RSA, GLUE, and the RoBERTa replication
notes/                          findings (chronological) and open items
tests/                          unit tests
figures/                        figures from the notes
archive/                        superseded code, kept for provenance
```

## Running

```bash
pip install -r requirements.txt

# BERT: RSR and matched control from one seed
python experiments/rsr_pretraining/bert_rsr_continued_pretraining.py \
    --regime compare --lambda 0.7 --seed 1

# GPT-2: same recipe, read out at layer 6
python experiments/rsr_pretraining/gpt2_rsr_continued_pretraining.py \
    --regime compare --lambda 0.7 --rsr-layer 6 --seed 1

# baselines and transfer, evaluation only
python experiments/rsr_pretraining/eval_retrofit_baseline.py
python experiments/rsr_pretraining/eval_postspec_baseline.py
python experiments/rsr_pretraining/eval_men_transfer.py

# tests
python -m unittest discover -s tests -t .
```

The sentence pool defaults to `artifacts/sentences_strict`, the full-Wikipedia
harvest every reported run used. It is built by
`experiments/contextual/harvest_sentences_strict.py` and
`filter_sentence_pool.py`. Corpora, the pool, checkpoints and result CSVs are
not versioned. Runners in `experiments/runners/` skip arms that have already
finished, so an interrupted batch resumes.

## Protocol

Each RSR arm has a control trained from the same seed on the same anchor
sentences with the RSR term removed. Reported numbers are RSR minus control,
averaged over seeds. The final model is reported. `--early-stop` keeps the
checkpoint with the best all-pairs SimLex instead, which selects on the test
set; the runs in `notes/findings.md` up to November 2026 used it.

## Switches

Both training scripts take these. Every one defaults to the configuration of
the reported runs, so a run with none of them reproduces those runs except for
early stopping.

| switch | effect |
|---|---|
| `--exclude-simlex-pairs` | drop every supervision pair that is also a SimLex pair (and its inflected copies) |
| `--inflect-verbs` | add third person, past and gerund copies of each SimVerb pair at the same score |
| `--things-pairs N` | train on a fixed random sample of N THINGS pairs instead of all 1.7M |
| `--sampler pairs` | build RSR batches from drawn scored pairs instead of uniform words |
| `--simverb-share F` | with `--sampler pairs`, draw a fraction F of pairs from SimVerb |
| `--split-sentences` | train on the first half of each word's sentences, evaluate on the second |
| `--early-stop` | keep the best all-pairs SimLex checkpoint |

`audit_supervision.py` takes the same supervision and sampler switches and
prints, without loading a model, what a configuration trains on: pairs per
source, how many SimLex pairs are in training, the SimLex split by part of
speech, and the scored pairs per RSR batch by source. Training logs the real
per-batch counts (`rsr_pairs_*` columns) and scores SimLex in finer splits as
well: Both split into pairs that were and were not training pairs, and every
partition by part of speech.

Inflected verb forms are new words with no harvested sentences. To use them in
the sentence context, list them and harvest them first:

```bash
python experiments/rsr_pretraining/audit_supervision.py --inflect-verbs \
    --write-missing artifacts/inflected_targets.txt
python -u experiments/contextual/harvest_sentences_strict.py \
    --targets-file artifacts/inflected_targets.txt
```
