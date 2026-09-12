"""Plot SimLex correlation against training step, averaged over seeds.

Each compare run writes an eval row every 100 steps for both arms, so the CSVs
already hold the whole trajectory - this just stacks them across seeds and draws
mean +/- standard error. Step 0 is the untouched checkpoint, so it's identical
in every run and gets drawn as the flat baseline.

Same layout as the original trajectory plots (one panel per partition) so the two
sets of figures can sit next to each other.

Note the read-out layer isn't in the CSV - it shows up as the step-0 value, which
is vanilla BERT at whatever layer the run evaluated (0.492 at layer 12, 0.523 at
layer 7). Pass --tag/--note so the two sets don't get mixed up.

    python experiments/rsr_pretraining/plot_training_curves.py 20260611_025638 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PARTITIONS = [
    ("all", "All pairs"),
    ("both", "Both in RSR"),
    ("one", "One in RSR"),
    ("neither", "Neither in RSR (generalisation)"),
]

# red for RSR to match the replication plots, blue for the control arm
RSR_C, MLM_C, VAN_C = "#d62728", "#1f77b4", "#555555"


def stack(stamps: list[str], arm: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (steps, {partition: seeds x steps array}) for one arm."""
    frames = [pd.read_csv(RESULTS / f"bert_continued_{arm}_{ts}.csv") for ts in stamps]
    n = min(len(f) for f in frames)          # in case a run stopped early
    steps = frames[0]["step"].to_numpy()[:n]
    cols = {key: np.vstack([f[f"{key}_rho"].to_numpy()[:n] for f in frames])
            for key, _ in PARTITIONS}
    return steps, cols


def band(ax, x, mat, colour, label):
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    ax.plot(x, mean, color=colour, marker="o", ms=3.5, lw=1.8, label=label, zorder=3)
    ax.fill_between(x, mean - sem, mean + sem, color=colour, alpha=0.18, lw=0)
    return mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stamps", nargs="+", help="run timestamps to average over")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--note", default="", help="extra text for the figure title")
    args = ap.parse_args()
    stamps = args.stamps

    steps, rsr = stack(stamps, "rsr")
    _, mlm = stack(stamps, "baseline")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (key, title) in zip(axes, PARTITIONS):
        vanilla = rsr[key][:, 0].mean()      # step 0, same checkpoint in both arms
        ax.axhline(vanilla, color=VAN_C, ls="--", lw=1.3, zorder=1,
                   label="Vanilla BERT (step 0)")
        band(ax, steps, mlm[key], MLM_C, "MLM control")
        mean = band(ax, steps, rsr[key], RSR_C, "RSR (weighted, lambda=0.7)")

        # ring the best trained checkpoint - skip step 0, that's the vanilla model
        # and on the Both panel it happens to be the highest point on the curve
        peak = 1 + int(np.argmax(mean[1:]))
        ax.plot(steps[peak], mean[peak], "o", ms=11, mfc="none", mec=RSR_C, mew=1.8,
                zorder=4)
        ax.annotate(f"best {mean[peak]:.3f} @ {steps[peak]}",
                    xy=(steps[peak], mean[peak]), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=8, color=RSR_C)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3, ls=":")
        ax.margins(y=0.16)
    axes[0].set_ylabel("SimLex-999 Spearman rho")
    axes[0].legend(fontsize=8.5, loc="lower left", framealpha=0.95)

    title = (f"Held-out SimLex-999 through training, mean +/- s.e. over "
             f"{len(stamps)} seeds")
    fig.suptitle(f"{title}{'  -  ' + args.note if args.note else ''}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIGURES / f"fig_training_curves{args.tag}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")

    # the numbers behind the rings, for the slide caption
    print(f"\n{'Partition':<34}{'vanilla':>9}{'MLM peak':>11}{'RSR peak':>11}{'RSR-MLM':>10}")
    for key, title in PARTITIONS:
        v = rsr[key][:, 0].mean()
        m, r = mlm[key].mean(axis=0), rsr[key].mean(axis=0)
        print(f"{title:<34}{v:>9.3f}{m.max():>11.3f}{r.max():>11.3f}"
              f"{r.max() - m.max():>+10.3f}")


if __name__ == "__main__":
    main()
