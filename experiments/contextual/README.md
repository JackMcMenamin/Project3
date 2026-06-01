# Contextual / sentence-level RSR pipeline

Newer line of work that addresses the "you used contextual models as static
encoders" critique: instead of encoding each word in isolation, sample up to
50 Wikipedia sentences per target, embed the target span in context, mean-pool
its subwords, and average across occurrences → one cached 768-d vector per
(model, target). That cached vector is used at **both** train and eval time.

## Stages (run in order)

| Stage | Script | What it does |
|-------|--------|--------------|
| 0a | `coverage_probe.py` | Count target occurrences in `data/AllCombined.txt` |
| 0b | `enwiki_fallback_probe.py` | Chase below-floor THINGS targets through the enwiki dump |
| A  | `harvest_sentences.py` | Harvest up to 50 sentences/target → `artifacts/sentences/<slug>.jsonl` (+ manifest). **Slow** (20+ min over enwiki) — don't rerun unless the target inventory or floor changes |
| B  | `extract_contextual_vectors.py --model {bert,gpt2}` | Cache 768-d vectors → `artifacts/cache/{model}/vectors.npz` |
| C  | `run_seeds_contextual.py --model {bert,gpt2}` | 10-seed RSR on cached vectors (transformer backbone not loaded; only the projection head trains) |
| — | `run_full_contextual.py` | Driver: both models, 10 seeds, writes the paper-ready summary |
| — | `tune_lr_epochs.py` | lr / epoch sweep for the contextual setup |

`run_full_contextual.py` and `tune_lr_epochs.py` import `run_seeds_contextual`
as a sibling module, so launch them from this directory (or with this folder on
the path).

## Status note
Word-level isolated encoding remains the headline result (`../transformers_word/`).
This pipeline is the in-flight methodological extension.
