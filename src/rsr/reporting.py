"""
Result reporting for the multi-seed experiments: write the per-seed Excel
sheet and print the aggregated summary the paper's tables are built from.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import paths

SUMMARY_METRICS = [
    ("All pairs - Vanilla", "vanilla_all"),
    ("All pairs - RSR", "rsr_all"),
    ("All pairs - Delta", "delta_all"),
    ("Both in RSR - Delta", "delta_both"),
    ("One in RSR - Delta", "delta_one"),
    ("Neither in RSR - Delta", "delta_neither"),
]


def save_seed_results(results: list[dict], model_label: str) -> Path:
    """Write `results/{model_label}_seeds_{timestamp}.xlsx`; return the path."""
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = paths.RESULTS_DIR / f"{model_label}_seeds_{timestamp}.xlsx"
    pd.DataFrame(results).to_excel(out, index=False)
    return out


def print_summary(results: list[dict], title: str) -> None:
    """Print per-seed rows, mean/std table, and the headline RSR findings."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(f"{title} - MULTI-SEED RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Seed':<6} {'Van All':>10} {'RSR All':>10} {'d All':>10} {'d Neither':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['seed']:<6} {r['vanilla_all']:>10.4f} {r['rsr_all']:>10.4f} "
              f"{r['delta_all']:>+10.4f} {r['delta_neither']:>+12.4f}")

    print("\nAggregated (Mean +/- Std):")
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for name, col in SUMMARY_METRICS:
        v = df[col].dropna()
        if len(v):
            print(f"{name:<30} {v.mean():>10.4f} {v.std():>10.4f} {v.min():>10.4f} {v.max():>10.4f}")

    delta_all = df["delta_all"].dropna()
    delta_neither = df["delta_neither"].dropna()
    print("\nKEY FINDINGS:")
    print(f"  Overall  d_rho = {delta_all.mean():+.4f} +/- {delta_all.std():.4f}")
    print(f"  Neither  d_rho = {delta_neither.mean():+.4f} +/- {delta_neither.std():.4f}  "
          f"(>0 in {(delta_neither > 0).sum()}/{len(delta_neither)} seeds)")
