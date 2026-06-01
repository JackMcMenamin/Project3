"""
Single-seed hyperparameter sweep for the contextual RSR setup.

Searches over (learning_rate, n_epochs) for BERT contextual to find a setting
where RSR meaningfully improves over the strong vanilla baseline (0.456).
The default 1e-3 / 200-epoch protocol overshoots and degrades held-out SimLex.

We expect a much lower LR + far fewer epochs to work better when starting
from rich pre-trained contextual representations.

Run with:
    python -u tune_lr_epochs.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import run_seeds_contextual as r

LR_GRID = [1e-4, 3e-4, 5e-4]
EPOCH_GRID = [50, 100]
SEED = 1
MODEL_NAME = "bert"
REPO_ROOT = Path(__file__).resolve().parents[2]  # experiments/contextual/ -> repo root
OUT_CSV = REPO_ROOT / "results" / f"tune_{MODEL_NAME}_seed{SEED}.csv"


def run_combo(vectors, all_pairs, rsr_words, simlex, lr: float, epochs: int) -> dict:
    """Run a single seed at the given (lr, epochs) and return summary metrics."""
    # Patch the module-level constants the run_single_seed call reads from.
    r.RSR_LR = lr
    r.RSR_EPOCHS = epochs

    print(f"\n--- lr={lr}, epochs={epochs} ---", flush=True)
    t0 = time.time()
    res = r.run_single_seed(SEED, vectors, all_pairs, rsr_words, simlex)
    res["lr"] = lr
    res["epochs"] = epochs
    res["elapsed_min"] = (time.time() - t0) / 60
    print(f"  combo done in {res['elapsed_min']:.1f} min  "
          f"(van={res['vanilla_all']:.4f} -> rsr={res['rsr_all']:.4f}, "
          f"delta_all={res['delta_all']:+.4f}, "
          f"delta_neither={res['delta_neither']:+.4f})", flush=True)
    return res


def main() -> None:
    print(f"=== Hyperparameter sweep — {MODEL_NAME}, seed {SEED} ===", flush=True)
    print(f"LR grid:    {LR_GRID}", flush=True)
    print(f"Epoch grid: {EPOCH_GRID}", flush=True)
    print(f"Total combos: {len(LR_GRID) * len(EPOCH_GRID)}", flush=True)

    print("\nLoading cache + datasets ...", flush=True)
    vectors = r.load_cached_vectors(MODEL_NAME)
    all_pairs, info, rsr_words = r.load_all_rsr_datasets()
    simlex = r.load_simlex(r.SIMLEX_PATH)
    print(f"  cache: {len(vectors)} keys, RSR pairs: {info['total']}, "
          f"SimLex pairs: {len(simlex)}", flush=True)

    rows = []
    overall_t0 = time.time()
    for lr in LR_GRID:
        for epochs in EPOCH_GRID:
            try:
                row = run_combo(vectors, all_pairs, rsr_words, simlex, lr, epochs)
                rows.append(row)
            except Exception as e:
                print(f"  combo lr={lr} epochs={epochs} FAILED: {e}", flush=True)

    print(f"\nSweep complete in {(time.time()-overall_t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults written to {OUT_CSV}", flush=True)

    # Print a compact ranking by delta_all.
    cols = ["lr", "epochs", "vanilla_all", "rsr_all",
            "delta_all", "delta_both", "delta_one", "delta_neither",
            "elapsed_min"]
    df_sorted = df[cols].sort_values("delta_all", ascending=False).reset_index(drop=True)
    print("\n=== Ranked by delta_all ===", flush=True)
    print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
