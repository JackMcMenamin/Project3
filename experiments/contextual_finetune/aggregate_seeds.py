"""Pull the multi-seed compare runs together into one table.

Takes run timestamps on the command line and pairs up the rsr/baseline CSVs for
each. Vanilla is the step-0 row; RSR and MLM are the best row by all_rho, which
is the checkpoint early stopping would have kept. Prints mean +/- std per
partition.

    python experiments/contextual_finetune/aggregate_seeds.py <ts1> <ts2> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"
PARTITIONS = ("all", "both", "one", "neither")


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["all_rho"].idxmax()]


def main() -> None:
    stamps = sys.argv[1:]
    if not stamps:
        sys.exit("usage: aggregate_seeds.py <timestamp> [<timestamp> ...]")

    rows = []
    for ts in stamps:
        rsr = pd.read_csv(RESULTS / f"bert_continued_rsr_{ts}.csv")
        base = pd.read_csv(RESULTS / f"bert_continued_baseline_{ts}.csv")
        van, rbest, bbest = rsr.iloc[0], best_row(rsr), best_row(base)
        row = {"ts": ts, "rsr_step": int(rbest["step"]), "mlm_step": int(bbest["step"])}
        for p in PARTITIONS:
            row[f"van_{p}"] = van[f"{p}_rho"]
            row[f"mlm_{p}"] = bbest[f"{p}_rho"]
            row[f"rsr_{p}"] = rbest[f"{p}_rho"]
        rows.append(row)
    df = pd.DataFrame(rows)

    print(f"\nSeeds aggregated: {len(df)}  (RSR best steps: {sorted(df['rsr_step'])}, "
          f"MLM best steps: {sorted(df['mlm_step'])})")
    print("=" * 76)
    print(f"{'Category':<16}{'Vanilla':>12}{'+MLM':>14}{'RSR':>14}{'RSR-Van':>10}{'RSR-MLM':>10}")
    print("-" * 76)
    label = {"all": "All pairs", "both": "Both in RSR",
             "one": "One in RSR", "neither": "Neither in RSR"}
    for p in PARTITIONS:
        v, m, r = df[f"van_{p}"], df[f"mlm_{p}"], df[f"rsr_{p}"]
        print(f"{label[p]:<16}"
              f"{v.mean():>7.3f}+/-{v.std():.3f}"
              f"{m.mean():>9.3f}+/-{m.std():.3f}"
              f"{r.mean():>9.3f}+/-{r.std():.3f}"
              f"{(r - v).mean():>+10.3f}"
              f"{(r - m).mean():>+10.3f}")

    out = RESULTS / f"seed_aggregate_{'_'.join(stamps[:1])}_n{len(df)}.csv"
    df.to_csv(out, index=False)
    print(f"\nPer-seed table saved to: {out}")


if __name__ == "__main__":
    main()
