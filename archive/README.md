# archive/ — superseded material (kept for provenance)

Nothing here is part of the current pipeline. Kept so results can be traced
back and old approaches re-examined.

- **transformers_word/** — **legacy** word-level transformer RSR (`run_seeds.py
  --model bert|gpt2`). Was the paper's Table 2; superseded by the sentence-level
  / continued-pre-training direction. Still runnable; kept as a comparison
  point. See its own README.
- **old_scripts/**
  - `run_bert_seeds.py`, `run_gpt2_seeds.py` — the original near-duplicate
    word-level runners, consolidated into `transformers_word/run_seeds.py`.
  - `BERT.py`, `GPT2.py` — earlier single-file versions the seed scripts were
    factored out of.
  - `run_seeds_*.py` (`distill`, `distill_v2`, `men`, `multi`, `propagate`,
    `run_seeds.py`) — exploratory RSR variants that didn't make the paper.
  - `seed_results_interleaved_sweep.xlsx` — stale results.
  - `Mark.txt` — **reference, not throwaway:** Mark Ormerod's RSR thesis
    chapter (Ch. 7). The methodological foundation for the new BERT
    continued-pre-training work; cited in
    `docs/bert_continued_pretraining_rsr_design.md`. (A copy also sits at repo
    root.)
- **notebooks/** — early exploratory notebooks (`gpt2`, `interleave`, `main`,
  `testing`).
- **brain_chapter/** — **reference, not throwaway:** Mark Ormerod's three
  analysis notebooks for the RSR/brain chapter. Source of the layer-freezing
  scheme (`embeddings + encoder.layer[:11]` frozen) and hyperparameters reused
  in the design note.
