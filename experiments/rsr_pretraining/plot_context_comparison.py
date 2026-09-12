"""The original four-panel layout, with the dimension that design cannot separate.

The original figure is four partitions x one evaluation context. Evaluation context is a
separate flag in the original code (eval_simlex.py, default "bare"), so a curve that moves
could be the training changing or the measurement changing, and the figure cannot
say which. Here every checkpoint is scored in three contexts, so the rows
separate them:

    row 1  sentences  each word averaged over 20 real sentences (ours)
    row 2  bare       the word alone, no sentence (the original default)
    row 3  template   the null frame "The word is X." (the original rsr_all training form)

Arms: RSR trained on real sentences, RSR trained on templates (the original rsr_all), and
the matched MLM control. Vanilla is the step-0 point of every curve, which is why
they all start together.

Usage:
    python experiments/rsr_pretraining/plot_context_comparison.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "fig_context_comparison.png")

CONTEXTS = [("sentences", ""), ("bare", "bare_"), ("template", "template_")]
PARTS = [("all", "All pairs"), ("both", "Both in RSR"),
         ("one", "One in RSR"), ("neither", "Neither in RSR")]
ARMS = [
    ("rsr_ctx_sentences", "RSR, real sentences", "#2A9D8F", "-"),
    ("rsr_ctx_template", "RSR, templates (the original rsr_all)", "#E8743B", "-"),
    ("baseline_ctx_sentences", "MLM control", "#5A6588", "--"),
]


def load() -> dict[str, list[pd.DataFrame]]:
    """Group every finished arm CSV by arm, one frame per seed."""
    runs: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RESULTS, "bert_continued_*ctx_*.csv"))):
        base = os.path.basename(path)
        m = re.match(r"bert_continued_(rsr|baseline)_(ctx_\w+?)_s(\d+)_\d{8}_\d{6}\.csv", base)
        if not m:
            continue
        regime, ctx, seed = m.groups()
        runs[f"{regime}_{ctx}"].append(pd.read_csv(path))
    return runs


def mean_curve(frames: list[pd.DataFrame], col: str):
    """Mean and range across seeds on the shared step grid."""
    if not frames or col not in frames[0].columns:
        return None, None, None
    steps = sorted(set.intersection(*[set(f["step"]) for f in frames]))
    vals = np.array([[f.loc[f.step == s, col].iloc[0] for s in steps] for f in frames])
    return np.array(steps), vals.mean(axis=0), (vals.min(axis=0), vals.max(axis=0))


def main() -> None:
    runs = load()
    if not runs:
        sys.exit("no ctx_* result CSVs found in results/")
    n_seeds = {k: len(v) for k, v in runs.items()}
    print("arms found:", n_seeds)

    fig, axes = plt.subplots(len(CONTEXTS), len(PARTS), figsize=(17, 10.5), sharex=True)
    for r, (ctx_name, pre) in enumerate(CONTEXTS):
        for c, (part, part_label) in enumerate(PARTS):
            ax = axes[r, c]
            col = f"{pre}{part}_rho"
            for key, label, colour, style in ARMS:
                frames = runs.get(key)
                if not frames:
                    continue
                steps, mean, rng = mean_curve(frames, col)
                if steps is None:
                    continue
                ax.plot(steps, mean, style, color=colour, lw=1.9,
                        label=label if (r == 0 and c == 0) else None)
                if rng is not None and len(frames) > 1:
                    ax.fill_between(steps, rng[0], rng[1], color=colour, alpha=0.15, lw=0)
            ax.grid(alpha=0.25, lw=0.6)
            if r == 0:
                ax.set_title(part_label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"eval: {ctx_name}\nSpearman rho", fontsize=10.5)
            if r == len(CONTEXTS) - 1:
                ax.set_xlabel("steps", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11.5)
    seeds = max(n_seeds.values()) if n_seeds else 0
    fig.suptitle("RSR under three evaluation contexts  "
                 f"(shaded band = range across {seeds} seeds)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)

    # the number that matters: RSR minus its matched control, per context
    print("\nRSR (sentences) minus MLM control, Neither partition:")
    print(f"{'step':>6}" + "".join(f"{c:>13}" for c, _ in CONTEXTS))
    rsr, ctl = runs.get("rsr_ctx_sentences"), runs.get("baseline_ctx_sentences")
    if rsr and ctl:
        grids = []
        for _, pre in CONTEXTS:
            s, m, _ = mean_curve(rsr, f"{pre}neither_rho")
            s2, m2, _ = mean_curve(ctl, f"{pre}neither_rho")
            grids.append((s, m - m2) if s is not None else (None, None))
        steps = grids[0][0]
        for i, s in enumerate(steps):
            if s % 400 == 0:
                print(f"{int(s):>6}" + "".join(f"{g[1][i]:>+13.4f}" for g in grids))


if __name__ == "__main__":
    main()
