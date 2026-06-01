# Word2Vec-from-scratch RSR

The static-embedding baseline: a skip-gram + negative-sampling Word2Vec trained
from scratch on Wikipedia, with and without the RSR loss.

| Script | What it does |
|--------|--------------|
| `main.py` | One end-to-end run: train vanilla + RSR Word2Vec, evaluate on SimLex-999. Notebook-style with `#---` section separators. |
| `run_seeds_interleaved.py` | 10-seed run with **interleaved** RSR (alternate pure W2V steps and pure RSR steps rather than weighting a combined loss); sweeps `RSR_FREQUENCY`. |

```bash
python main.py
python run_seeds_interleaved.py
```

These were the first RSR experiments; the key finding is that gains stay
concentrated in the "both words supervised" partition and barely spread to
unsupervised pairs — unlike the pretrained transformers (`../transformers_word/`).
