# Word-level transformer RSR  — LEGACY (archived 2026-06-01)

> **Superseded.** This was the headline BERT/GPT-2 result in the early draft of
> the paper (Table 2), but the project has moved to sentence-level / contextual
> training. Kept for provenance and as a comparison point. Not the current line
> of work.

Words are encoded **in isolation** (no surrounding context); a mostly-frozen
transformer feeds a 768→128 projection head, which is the only thing RSR trains.

```bash
# still runnable from anywhere (paths anchor to repo root):
python archive/transformers_word/run_seeds.py --model bert
python archive/transformers_word/run_seeds.py --model gpt2
```

Shared logic lives in `src/rsr/`; this file is just config + orchestration. It
consolidated the older near-duplicate `run_bert_seeds.py` / `run_gpt2_seeds.py`
(in `../old_scripts/`).

## What replaced it
- **Frozen contextual vectors:** `experiments/contextual/` — words embedded in
  real sentences, pooled into cached vectors (a stronger but still
  static-extraction approach).
- **Continued pre-training (current direction):** see
  `docs/bert_continued_pretraining_rsr_design.md` — BERT's own weights train via
  interleaved MLM + RSR, answering the "you're just gleaning embeddings"
  critique.
