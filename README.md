# Project 3 — Representational Similarity Regularisation (RSR)

Experiment 3 of the PhD: **RSR** adds a rank-based similarity loss
(differentiable Spearman, `L_rsr = 1 − ρ_soft(ŝ, s)`) over the cosine
similarities of a model's embeddings, supervised by pooled human similarity
judgements (**MEN + SimVerb-3500 + THINGS**) and evaluated on held-out
**SimLex-999**, partitioned by how many of each pair's words were seen in the
RSR supervision set (*both / one / neither in RSR*).

The full write-up is Appendix B of `docs/differentiation_report.pdf`.

---

## Repository layout

```
.
├── src/rsr/                  Shared library (one source of truth)
│   ├── paths.py              All on-disk paths, resolved from the repo root
│   ├── seeds.py              set_seed
│   ├── losses.py             soft_spearman / soft_rank (the RSR loss)
│   ├── datasets.py           Load MEN / SimVerb / THINGS / SimLex-999
│   ├── models.py             BERT & GPT-2 word-embedding wrappers
│   ├── train_eval.py         train_rsr, evaluate_simlex, run_single_seed
│   └── reporting.py          Excel + summary output
│
├── experiments/              Runnable entry points (thin; import from src/rsr)
│   ├── word2vec/             Word2Vec-from-scratch RSR (paper Table 1)
│   │   ├── main.py             single-run vanilla vs RSR
│   │   └── run_seeds_interleaved.py   10-seed interleaved sweep
│   ├── contextual/           ★ Sentence-level / contextual-vector pipeline
│   │   ├── coverage_probe.py            Stage 0a — corpus coverage
│   │   ├── enwiki_fallback_probe.py     Stage 0b — enwiki fallback coverage
│   │   ├── harvest_sentences.py         Stage A — sentence harvest
│   │   ├── extract_contextual_vectors.py Stage B — cached 768-d vectors
│   │   ├── run_seeds_contextual.py      Stage C — 10-seed RSR on cached vectors
│   │   ├── run_full_contextual.py       full BERT+GPT-2 driver
│   │   └── tune_lr_epochs.py            lr/epoch sweep
│   └── visualize_results.ipynb   Figures for the paper
│
│   (NEXT: BERT continued pre-training + interleaved MLM/RSR — see
│    docs/bert_continued_pretraining_rsr_design.md. Not yet implemented.)
│
├── data/                     Corpora & eval sets (gitignored except SimLex-999)
│   ├── SimLex-999/  SICK/  STS_Benchmark/  MEN/  simverb-3500-data/
│   ├── AllCombined.txt  enwiki_namespace_0/  ...   (large, not versioned)
├── things_similarity/        THINGS SPoSE bundle (external, not versioned)
│
├── models/  results/  figures/   Trained checkpoints, Excel results, plots
├── artifacts/                Generated pipeline outputs (gitignored):
│   └── cache/  sentences/  coverage_probe*.csv  sentences_manifest.csv
├── docs/                     Paper PDF, design notes (see docs/ section below)
│
└── archive/                  Superseded code, kept for provenance
    ├── transformers_word/      LEGACY word-level BERT/GPT-2 RSR (old Table 2)
    ├── old_scripts/            originals incl. run_bert_seeds.py / run_gpt2_seeds.py, Mark.txt
    ├── notebooks/              old exploratory notebooks
    └── brain_chapter/          Mark Ormerod's RSR thesis notebooks (reference)
```

★ = the experiment currently being actively worked on. Word-level transformer
RSR is now **legacy** (moved to `archive/transformers_word/`).

---

## Setup

```bash
pip install torch transformers numpy pandas scipy scikit-learn tqdm openpyxl
# optional but recommended (faster, lower-memory RSR loss):
pip install torchsort
# contextual pipeline only:
pip install pyahocorasick nltk
```

Paths are anchored to the repo root via `src/rsr/paths.py`, so the scripts run
from anywhere. The two corpus-heavy stages expect `data/AllCombined.txt` and
`data/enwiki_namespace_0/` to exist.

---

## Running the experiments

### Contextual / sentence-level pipeline (current focus)
Run the stages in order (Stage A is slow — see notes in each file):
```bash
python -u experiments/contextual/coverage_probe.py
python -u experiments/contextual/enwiki_fallback_probe.py
python -u experiments/contextual/harvest_sentences.py
python -u experiments/contextual/extract_contextual_vectors.py --model bert
python -u experiments/contextual/extract_contextual_vectors.py --model gpt2
python -u experiments/contextual/run_full_contextual.py
```
Words are embedded inside real Wikipedia sentences, pooled into cached vectors;
a projection head is RSR-trained on those vectors. The **next** step (BERT's own
weights training via interleaved MLM + RSR) is specified in
`docs/bert_continued_pretraining_rsr_design.md` and not yet implemented.

### Word2Vec RSR
```bash
python experiments/word2vec/main.py                  # one vanilla vs RSR run
python experiments/word2vec/run_seeds_interleaved.py # 10-seed interleaved sweep
```

### Word-level transformer RSR (LEGACY — archived)
The original headline transformer result (paper Table 2), kept for comparison.
Words encoded in isolation by a mostly-frozen transformer + 768→128 head.
```bash
python archive/transformers_word/run_seeds.py --model bert
python archive/transformers_word/run_seeds.py --model gpt2
```

---

## docs/

- `differentiation_report.pdf` — the wider PhD differentiation report (RSR is
  Experiment 3 / Appendix B).
- `bert_continued_pretraining_rsr_design.md` — design note for the next step:
  BERT continued pre-training with interleaved MLM + RSR (the answer to "you're
  just gleaning static embeddings"). Grounded in Mark Ormerod's RSR thesis
  chapter (`archive/old_scripts/Mark.txt`, notebooks in `archive/brain_chapter/`).
- The paper draft itself (`RSR_Paper__Copy_.pdf`) currently lives at repo root.

---

## The 10-seed contract

When changing the transformer experiments, keep the protocol the paper reports:
**10 seeds, vanilla vs RSR, SimLex-999 evaluation partitioned by
both/one/neither-in-RSR overlap.** The shared logic in `src/rsr/` exists so
this contract is defined in exactly one place — change it deliberately.
