"""
Word-level transformer RSR — multi-seed experiment (BERT or GPT-2).

This is the ACTIVE word-level RSR experiment. Each word is encoded in
isolation by a (mostly frozen) transformer, a small projection head is
RSR-trained against pooled human similarity judgements, and both the vanilla
and RSR models are evaluated on held-out SimLex-999, partitioned by RSR
supervision overlap.

It replaces the old, near-identical `run_bert_seeds.py` and
`run_gpt2_seeds.py` (now in `archive/old_scripts/`): the shared logic lives in
`src/rsr/`, and the only per-architecture differences are the model wrapper
(selected by `--model`) and the labels.

Usage:
    python experiments/transformers_word/run_seeds.py --model bert
    python experiments/transformers_word/run_seeds.py --model gpt2
    python experiments/transformers_word/run_seeds.py --model bert --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import rsr` work no matter where this is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rsr import datasets, reporting          # noqa: E402
from rsr.models import DEVICE, WRAPPERS       # noqa: E402
from rsr.losses import HAS_TORCHSORT          # noqa: E402
from rsr.train_eval import run_single_seed    # noqa: E402

# --- hyperparameters (the experimental contract — change deliberately) ------
RSR_EPOCHS = 200
RSR_LR = 1e-3
RSR_SAMPLE_SIZE = 10000
PROJECTION_DIM = 128
BATCH_SIZE = 64
DEFAULT_SEEDS = list(range(1, 11))


def main() -> None:
    ap = argparse.ArgumentParser(description="Word-level transformer RSR multi-seed run.")
    ap.add_argument("--model", choices=sorted(WRAPPERS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = ap.parse_args()

    label = args.model
    print("=" * 70)
    print(f"{label.upper()} RSR Multi-Seed Experiment")
    print(f"Seeds: {args.seeds}  Device: {DEVICE}  Torchsort: {HAS_TORCHSORT}")
    print("=" * 70)

    print("\nLoading datasets...")
    all_pairs, info, rsr_words = datasets.load_all_rsr_datasets()
    simlex = datasets.load_simlex()
    print(f"  Total RSR pairs: {info['total']}  RSR vocab: {info['unique_words']}  "
          f"SimLex pairs: {len(simlex)}")

    wrapper_cls = WRAPPERS[args.model]

    def model_factory():
        return wrapper_cls(projection_dim=PROJECTION_DIM, num_frozen_layers=11)

    results = [
        run_single_seed(seed, model_factory, all_pairs, rsr_words, simlex, hp=sys.modules[__name__])
        for seed in args.seeds
    ]

    out = reporting.save_seed_results(results, label)
    print(f"\nResults saved to: {out}")
    reporting.print_summary(results, label.upper())
    print("\nDone!")


if __name__ == "__main__":
    main()
