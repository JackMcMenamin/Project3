# Word-level transformer RSR (superseded)

Each word is encoded in isolation, a mostly frozen transformer feeds a
768 to 128 projection head, and RSR trains only that head. This was the first
BERT and GPT-2 result. It was replaced first by cached contextual vectors
(`experiments/contextual/`) and then by continued pre-training
(`experiments/rsr_pretraining/`), where the backbone itself trains.

```bash
python archive/transformers_word/run_seeds.py --model bert
python archive/transformers_word/run_seeds.py --model gpt2
```

The shared code is in `rsr/` next to this file.
