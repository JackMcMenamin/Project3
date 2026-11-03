# archive

Superseded code, kept so older results in `notes/findings.md` can be traced
back. Nothing here is part of the current pipeline.

- `transformers_word/` is the word-level transformer RSR, where each word is
  encoded on its own and only a projection head trains. It was the first
  BERT/GPT-2 result and was replaced by continued pre-training. Its shared
  library is `transformers_word/rsr/`.
- `old_scripts/` holds the near-duplicate runners that `transformers_word`
  consolidated, the single-file `BERT.py` and `GPT2.py` they came from, a set of
  exploratory variants (distillation, propagation, multi-dataset, MEN), and the
  multi-layer comparison runner whose flags were removed from the main script.
  The removed options are still available at the `layer-experiments` tag.
- `notebooks/` are early exploratory notebooks, outputs stripped.
