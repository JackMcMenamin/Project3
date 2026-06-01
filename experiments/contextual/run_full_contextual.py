"""
Full 10-seed contextual RSR runs for BERT and GPT-2, sequentially.

Uses the hyperparameters chosen by the LR/epochs sweep:
    lr=1e-4, epochs=100

Saves three artifacts:
    results/bert_contextual_seeds_{timestamp}.xlsx
    results/gpt2_contextual_seeds_{timestamp}.xlsx
    results/contextual_summary_{timestamp}.txt   (paper-ready Table 2)

Run with:
    python -u run_full_contextual.py
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import run_seeds_contextual as r

ROOT = Path(__file__).resolve().parents[2]  # experiments/contextual/ -> repo root
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LR = 1e-4
EPOCHS = 100
SEEDS = list(range(1, 11))


def run_one_model(model_name: str, timestamp: str) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 70, flush=True)
    print(f" {model_name.upper()} — 10 seeds, lr={LR}, epochs={EPOCHS}", flush=True)
    print("=" * 70, flush=True)

    # Patch module constants so run_single_seed picks them up.
    r.SEEDS = SEEDS
    r.RSR_LR = LR
    r.RSR_EPOCHS = EPOCHS

    print(f"Loading cached vectors for {model_name} ...", flush=True)
    vectors = r.load_cached_vectors(model_name)
    all_pairs, info, rsr_words = r.load_all_rsr_datasets()
    simlex = r.load_simlex(r.SIMLEX_PATH)

    print(f"  cache: {len(vectors)} keys", flush=True)
    print(f"  RSR pairs: {info['total']:,} (vocab {info['unique_words']})", flush=True)
    print(f"  SimLex pairs: {len(simlex)}", flush=True)

    rsr_in = sum(1 for w in rsr_words if w in vectors)
    sl_words = {w for p in simlex for w in p[:2]}
    sl_in = sum(1 for w in sl_words if w in vectors)
    print(f"  RSR vocab in cache: {rsr_in}/{len(rsr_words)} "
          f"({100*rsr_in/len(rsr_words):.1f}%)", flush=True)
    print(f"  SimLex vocab in cache: {sl_in}/{len(sl_words)} "
          f"({100*sl_in/len(sl_words):.1f}%)", flush=True)

    rows = []
    overall_t0 = time.time()
    for seed in SEEDS:
        seed_t0 = time.time()
        result = r.run_single_seed(seed, vectors, all_pairs, rsr_words, simlex)
        result["elapsed_min"] = (time.time() - seed_t0) / 60
        rows.append(result)
        print(f"  >>> seed {seed} done in {result['elapsed_min']:.1f} min, "
              f"running total {(time.time()-overall_t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{model_name}_contextual_seeds_{timestamp}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\nWrote per-seed Excel to {out_path}", flush=True)

    summary = {
        "model": model_name,
        "n_seeds": len(SEEDS),
        "vanilla_all_mean": df["vanilla_all"].mean(),
        "vanilla_all_std": df["vanilla_all"].std(),
        "rsr_all_mean": df["rsr_all"].mean(),
        "rsr_all_std": df["rsr_all"].std(),
        "delta_all_mean": df["delta_all"].mean(),
        "delta_all_std": df["delta_all"].std(),
        "delta_both_mean": df["delta_both"].mean(),
        "delta_both_std": df["delta_both"].std(),
        "delta_one_mean": df["delta_one"].mean(),
        "delta_one_std": df["delta_one"].std(),
        "delta_neither_mean": df["delta_neither"].mean(),
        "delta_neither_std": df["delta_neither"].std(),
        "vanilla_both_mean": df["vanilla_both"].mean(),
        "vanilla_one_mean": df["vanilla_one"].mean(),
        "vanilla_neither_mean": df["vanilla_neither"].mean(),
        "rsr_both_mean": df["rsr_both"].mean(),
        "rsr_one_mean": df["rsr_one"].mean(),
        "rsr_neither_mean": df["rsr_neither"].mean(),
        "n_simlex_evaluated": int(df["n_simlex_evaluated"].iloc[0]),
        "n_simlex_skipped": int(df["n_simlex_skipped"].iloc[0]),
        "elapsed_min": (time.time() - overall_t0) / 60,
    }
    return df, summary


def write_paper_summary(bert_summary: dict, gpt2_summary: dict, path: Path) -> None:
    """Write a paper-ready text summary in the same shape as the appendix Table 2."""
    lines: list[str] = []

    def add(s: str = "") -> None:
        lines.append(s)

    add("=" * 78)
    add(" CONTEXTUAL RSR — 10-seed results (lr=1e-4, epochs=100)")
    add(f" Generated: {datetime.now().isoformat(timespec='seconds')}")
    add("=" * 78)
    add()
    add("Hyperparameters identical to the original paper EXCEPT:")
    add(f"  - Word representations: cached contextual vectors")
    add(f"    (mean-pooled over up to 50 Wikipedia sentence occurrences)")
    add(f"  - Backbones fully frozen; only the 768->128 projection head trains")
    add(f"  - Learning rate: 1e-4 (was 1e-3)  [retuned for stronger baseline]")
    add(f"  - Epochs:        100 (was 200)   [retuned for stronger baseline]")
    add()
    add(f"SimLex-999 evaluation: {bert_summary['n_simlex_evaluated']} pairs "
        f"(skipped {bert_summary['n_simlex_skipped']} for cache-missing words)")
    add()

    # Headline table — same shape as Table 2 in the paper.
    add("-" * 78)
    add(" Table 2 (contextual): mean +/- std over 10 seeds")
    add("-" * 78)
    add(f"{'':<24}{'BERT':>22}{'GPT-2':>22}")
    add(f"{'Category':<24}{'Vanilla':>11}{'RSR':>11}{'Vanilla':>11}{'RSR':>11}")
    add("-" * 78)
    for label, vkey, rkey in [
        ("All pairs",        "vanilla_all_mean",     "rsr_all_mean"),
        ("Both in RSR",      "vanilla_both_mean",    "rsr_both_mean"),
        ("One in RSR",       "vanilla_one_mean",     "rsr_one_mean"),
        ("Neither in RSR",   "vanilla_neither_mean", "rsr_neither_mean"),
    ]:
        bv, br = bert_summary[vkey], bert_summary[rkey]
        gv, gr = gpt2_summary[vkey], gpt2_summary[rkey]
        add(f"{label:<24}{bv:>11.3f}{br:>11.3f}{gv:>11.3f}{gr:>11.3f}")
    add("-" * 78)
    add()

    # Delta table with std.
    add("-" * 78)
    add(" Deltas (RSR - Vanilla), mean +/- std over 10 seeds")
    add("-" * 78)
    add(f"{'Category':<24}{'BERT':>26}{'GPT-2':>26}")
    add("-" * 78)
    for label, key in [
        ("All pairs",       "delta_all"),
        ("Both in RSR",     "delta_both"),
        ("One in RSR",      "delta_one"),
        ("Neither in RSR",  "delta_neither"),
    ]:
        bm, bs = bert_summary[f"{key}_mean"], bert_summary[f"{key}_std"]
        gm, gs = gpt2_summary[f"{key}_mean"], gpt2_summary[f"{key}_std"]
        add(f"{label:<24}{bm:>+12.4f} +/- {bs:.4f}{gm:>+12.4f} +/- {gs:.4f}")
    add("-" * 78)
    add()

    # Compact comparison vs the original paper.
    add("-" * 78)
    add(" Comparison with original paper (isolated-word encoding)")
    add("-" * 78)
    add(f"{'':<14}{'Vanilla':>12}{'RSR':>10}{'Delta':>10}{'   Notes':<32}")
    add("-" * 78)
    add(f"{'BERT (orig)':<14}{0.148:>12.3f}{0.281:>10.3f}{0.133:>+10.3f}"
        f"   isolated-word, lr=1e-3, 200ep")
    add(f"{'BERT (ctx)':<14}{bert_summary['vanilla_all_mean']:>12.3f}"
        f"{bert_summary['rsr_all_mean']:>10.3f}"
        f"{bert_summary['delta_all_mean']:>+10.3f}"
        f"   contextual, lr=1e-4, 100ep")
    add(f"{'GPT-2 (orig)':<14}{0.097:>12.3f}{0.226:>10.3f}{0.129:>+10.3f}"
        f"   isolated-word, lr=1e-3, 200ep")
    add(f"{'GPT-2 (ctx)':<14}{gpt2_summary['vanilla_all_mean']:>12.3f}"
        f"{gpt2_summary['rsr_all_mean']:>10.3f}"
        f"{gpt2_summary['delta_all_mean']:>+10.3f}"
        f"   contextual, lr=1e-4, 100ep")
    add("-" * 78)
    add()
    add("=" * 78)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Paper-ready summary written to {path}", flush=True)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_t0 = time.time()

    print(f"=== Full contextual RSR run — {timestamp} ===", flush=True)
    print(f"Models: bert, gpt2", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Hyperparams: lr={LR}, epochs={EPOCHS}", flush=True)

    bert_df, bert_summary = run_one_model("bert", timestamp)
    gpt2_df, gpt2_summary = run_one_model("gpt2", timestamp)

    summary_path = RESULTS_DIR / f"contextual_summary_{timestamp}.txt"
    write_paper_summary(bert_summary, gpt2_summary, summary_path)

    print(f"\nFull run complete in {(time.time() - overall_t0)/60:.1f} min", flush=True)
    print(f"  BERT:    {bert_summary['elapsed_min']:.1f} min", flush=True)
    print(f"  GPT-2:   {gpt2_summary['elapsed_min']:.1f} min", flush=True)
    print("\nFinal headline:", flush=True)
    print(f"  BERT  vanilla {bert_summary['vanilla_all_mean']:.3f} -> "
          f"RSR {bert_summary['rsr_all_mean']:.3f} "
          f"(d={bert_summary['delta_all_mean']:+.4f} +/- {bert_summary['delta_all_std']:.4f})",
          flush=True)
    print(f"  GPT-2 vanilla {gpt2_summary['vanilla_all_mean']:.3f} -> "
          f"RSR {gpt2_summary['rsr_all_mean']:.3f} "
          f"(d={gpt2_summary['delta_all_mean']:+.4f} +/- {gpt2_summary['delta_all_std']:.4f})",
          flush=True)


if __name__ == "__main__":
    main()
