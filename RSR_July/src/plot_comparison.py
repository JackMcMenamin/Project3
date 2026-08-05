import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot cross-run comparison")
    parser.add_argument("--old_dir", type=str, default="results_bdhomepc_2026-07-01")
    parser.add_argument("--new_dir", type=str, default="results_bdhomepc_2026-07-03")
    parser.add_argument("--out_dir", type=str, default="results_comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.old_dir, "all_trajectories_data.json"), 'r') as f:
        data_old = json.load(f)
    with open(os.path.join(args.new_dir, "all_trajectories_data.json"), 'r') as f:
        data_new = json.load(f)

    # We map old mode names to new mode names
    mode_map = {
        "vanilla_all": "vanilla_all",
        "vanilla_targetwords": "vanilla_targetwords",
        "rsr_all": "rsr_all",
        "rsr_targetwords": "rsr_targetwords"
    }
    
    colors = {
        "vanilla_all": "#3B82F6",
        "vanilla_targetwords": "#10B981",
        "rsr_all": "#EF4444",
        "rsr_targetwords": "#F59E0B"
    }

    spearman_cats = ["All pairs", "Both in RSR", "One in RSR", "Neither in RSR"]
    pearson_cats = ["All pairs (Pearson)", "Both in RSR (Pearson)", "One in RSR (Pearson)", "Neither in RSR (Pearson)"]

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])
    fig.suptitle("Cross-Run Overlay: 2026-07-01 (Solid) vs 2026-07-03 (Dashed)", fontsize=18, fontweight='bold', y=0.98)

    axs_spearman = []
    for j, cat in enumerate(spearman_cats):
        ax = fig.add_subplot(gs[0, j])
        axs_spearman.append(ax)

        for old_mode, new_mode in mode_map.items():
            color = colors[old_mode]
            
            # Old data
            if old_mode in data_old:
                steps_old = data_old[old_mode]["steps"]
                mean_old = np.array(data_old[old_mode]["scores"][cat]["mean"], dtype=np.float64)
                ax.plot(steps_old, mean_old, color=color, linestyle='solid', linewidth=2.0, label=f"{old_mode} (07-01)")

            # New data
            if new_mode in data_new:
                steps_new = data_new[new_mode]["steps"]
                mean_new = np.array(data_new[new_mode]["scores"][cat]["mean"], dtype=np.float64)
                ax.plot(steps_new, mean_new, color=color, linestyle='dashed', linewidth=2.0, marker='o', markersize=3, label=f"{old_mode} (07-03)")

        ax.set_title(f"{cat} (Spearman rho)", fontsize=11, fontweight='semibold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("rho", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        if j == 0:
            ax.legend(loc='best', fontsize=8)

    axs_pearson = []
    for j, cat in enumerate(pearson_cats):
        ax = fig.add_subplot(gs[1, j], sharex=axs_spearman[j])
        axs_pearson.append(ax)

        for old_mode, new_mode in mode_map.items():
            color = colors[old_mode]
            if old_mode in data_old:
                ax.plot(data_old[old_mode]["steps"], data_old[old_mode]["scores"][cat]["mean"], color=color, linestyle='solid', linewidth=2.0)
            if new_mode in data_new:
                ax.plot(data_new[new_mode]["steps"], data_new[new_mode]["scores"][cat]["mean"], color=color, linestyle='dashed', linewidth=2.0, marker='o', markersize=3)
        
        ax.set_title(f"{cat} (Pearson r)", fontsize=11, fontweight='semibold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("r", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # MLM Loss
    ax_loss = fig.add_subplot(gs[2, :])
    for old_mode, new_mode in mode_map.items():
        color = colors[old_mode]
        if old_mode in data_old:
            steps_old = np.array(data_old[old_mode]["steps"])
            mean_old = np.array(data_old[old_mode]["scores"]["MLM Loss"]["mean"], dtype=np.float64)
            mask = ~np.isnan(mean_old)
            if np.any(mask):
                ax_loss.plot(steps_old[mask], mean_old[mask], color=color, linestyle='solid', linewidth=2.0, label=f"{old_mode} (07-01)")
        
        if new_mode in data_new:
            steps_new = np.array(data_new[new_mode]["steps"])
            mean_new = np.array(data_new[new_mode]["scores"]["MLM Loss"]["mean"], dtype=np.float64)
            mask = ~np.isnan(mean_new)
            if np.any(mask):
                ax_loss.plot(steps_new[mask], mean_new[mask], color=color, linestyle='dashed', marker='^', markersize=4, linewidth=2.0, label=f"{old_mode} (07-03)")

    ax_loss.set_title("MLM Training Loss Curve", fontsize=13, fontweight='bold')
    ax_loss.set_xlabel("Training Steps", fontsize=11)
    ax_loss.set_ylabel("Loss (Cross Entropy)", fontsize=11)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(args.out_dir, "overlay_01_vs_03_final.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Comparison overlay plot saved to {out_path}")

if __name__ == "__main__":
    main()
