# Sentence pool and cached contextual vectors

This folder builds the Wikipedia sentence pool every continued pre-training run
uses: `harvest_sentences_strict.py`, then `filter_sentence_pool.py`, writing
`artifacts/sentences_strict/<word>.jsonl`.

The other stages are the earlier cached-vector approach. Each target word was
embedded in up to 50 sentences, pooled to one 768-d vector, cached, and a
projection head was trained on the cache. They are kept because the notes
report their results.

| Stage | Script | Purpose |
|-------|--------|---------|
| 0a | `coverage_probe.py` | count target occurrences in the corpus |
| 0b | `enwiki_fallback_probe.py` | find rare THINGS targets in the full enwiki dump |
| A | `harvest_sentences.py` | first harvest, Simple English pool (slow) |
| A' | `harvest_sentences_strict.py`, `filter_sentence_pool.py` | full-Wikipedia pool used by all reported runs |
| B | `extract_contextual_vectors.py --model {bert,gpt2}` | cache one vector per word |
| C | `run_seeds_contextual.py --model {bert,gpt2}` | multi-seed RSR on the cached vectors |
| | `run_full_contextual.py` | both models, ten seeds, summary table |
| | `tune_lr_epochs.py` | learning-rate and epoch sweep |
| | `audit_harvest_quality.py` | spot checks on harvested sentences |

`run_full_contextual.py` and `tune_lr_epochs.py` import `run_seeds_contextual`
as a sibling, so run them from this folder.
