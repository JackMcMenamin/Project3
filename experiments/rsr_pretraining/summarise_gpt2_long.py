"""Summarise the GPT-2 extended-training runs in the BERT table's format.

Reads every completed results/gpt2_continued_{rsr,baseline}_long_s*.csv pair
(and falls back to the partial_ file for a seed still in progress, flagged as
such), computes RSR minus matched control at every evaluation step, and reports
for each partition the peak delta and its step, the endpoint delta, and the mean
over the last 2000 steps, as mean [min, max] across seeds.

Reported at matched steps, not early-stopped, because these runs disable early
stopping: on GPT-2 the all-pairs score declines, so a best-all-pairs rule would
return the untrained model.

Usage:
    python experiments/rsr_pretraining/summarise_gpt2_long.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
PARTS = [("neither", "Neither in RSR"), ("all", "All pairs"),
         ("one", "One in RSR"), ("both", "Both in RSR")]

# BERT extended run (8000 steps, 1e-5, 3 seeds), from notes/findings.md, for the
# side-by-side column.
BERT = {"neither": ("+0.099 @1200", "+0.065", "+0.072"),
        "all":     ("+0.047 @800",  "-0.002", "+0.005"),
        "both":    ("n/a",          "-0.062", "-0.061"),
        "one":     ("n/a",          "n/a",    "n/a")}


def load(regime, seed):
    done = sorted(glob.glob(os.path.join(RES, f"gpt2_continued_{regime}_long_s{seed}_*.csv")))
    if done:
        return pd.read_csv(done[-1]), True
    part = os.path.join(RES, f"partial_gpt2_{regime}_long_s{seed}.csv")
    if os.path.exists(part):
        return pd.read_csv(part), False
    return None, False


seeds = sorted({int(m.group(1)) for f in glob.glob(os.path.join(RES, "*gpt2_*_long_s*.csv"))
                for m in [re.search(r"long_s(\d+)", f)] if m})
per_seed = {}
for s in seeds:
    r, rf = load("rsr", s)
    b, bf = load("baseline", s)
    if r is None or b is None:
        continue
    m = r.merge(b, on="step", suffixes=("", "_c"))
    for p, _ in PARTS:
        m[f"d_{p}"] = m[f"{p}_rho"] - m[f"{p}_rho_c"]
    per_seed[s] = (m, rf and bf)

if not per_seed:
    raise SystemExit("no long_s* runs with both arms found")

print("GPT-2 extended training, 8000 steps @ 1e-5, no early stopping, RSR minus control")
print("seeds:", ", ".join(f"{s}{'' if c else ' (partial)'}" for s, (_, c) in per_seed.items()))
print()

def agg(vals):
    v = np.array(vals)
    return f"{v.mean():+.3f} [{v.min():+.3f}, {v.max():+.3f}]" if len(v) > 1 else f"{v[0]:+.3f}"

print(f"{'partition':<16}{'peak (step)':>22}{'at end':>26}{'mean last 2000':>26}   BERT: peak / end / last2k")
for p, label in PARTS:
    peaks, pk_steps, ends, tails = [], [], [], []
    for s, (m, _) in per_seed.items():
        d = m[f"d_{p}"].values
        i = int(np.argmax(d)); peaks.append(d[i]); pk_steps.append(int(m.step.iloc[i]))
        ends.append(d[-1])
        last = m[m.step > m.step.max() - 2000]
        tails.append(last[f"d_{p}"].mean())
    steps = sorted(set(pk_steps))
    pk = agg(peaks) + " @" + ("/".join(map(str, steps)) if len(steps) > 1 else str(steps[0]))
    print(f"{label:<16}{pk:>22}{agg(ends):>26}{agg(tails):>26}   {' / '.join(BERT[p])}")

print()
print("per-step Neither delta, mean across seeds:")
steps = per_seed[seeds[0]][0].step.values
for st in steps:
    vals = [m.loc[m.step == st, "d_neither"].iloc[0] for m, _ in per_seed.values()
            if (m.step == st).any()]
    both = [m.loc[m.step == st, "d_both"].iloc[0] for m, _ in per_seed.values()
            if (m.step == st).any()]
    if st % 800 == 0:
        print(f"  step {int(st):>5}   neither {np.mean(vals):+.4f}   both {np.mean(both):+.4f}")
