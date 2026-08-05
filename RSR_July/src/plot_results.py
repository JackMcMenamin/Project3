"""
Visualization and Aggregation Module for RSR Experiments

This script is responsible for parsing the live evaluation logs generated during training 
and converting them into a clean, matplotlib-based visual representation of learning trajectories.
It automatically calculates standard variance across multiple statistical replicates.
It plots Spearman rho, Pearson r, and MLM training loss.

Workflow Context:
- The final step in the experimental pipeline.
- Consumes: The `results/eval_log_{mode}_run{id}.txt` files outputted by `train.py`.
- Outputs: Multiple `results/*.png` graphs and data serialization files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

def parse_log_file(filepath):
    steps = []
    scores = {
        "MLM Loss": [],
        "All pairs": [],
        "Both in RSR": [],
        "One in RSR": [],
        "Neither in RSR": [],
        "All pairs (Pearson)": [],
        "Both in RSR (Pearson)": [],
        "One in RSR (Pearson)": [],
        "Neither in RSR (Pearson)": []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Step"):
                parts = line.split("|")
                step_str = parts[0].strip().replace("Step", "").strip()
                try:
                    step = int(step_str)
                except ValueError:
                    continue
                steps.append(step)
                
                # Temp dict to store parsed values for this line
                row_vals = {k: None for k in scores.keys()}
                
                for part in parts[1:]:
                    if ":" in part:
                        cat, score_str = part.split(":", 1)
                        cat = cat.strip()
                        score_str = score_str.strip()
                        try:
                            score = float(score_str)
                        except ValueError:
                            score = None
                        if cat in row_vals:
                            row_vals[cat] = score
                            
                for cat in scores.keys():
                    scores[cat].append(row_vals[cat])
                    
    return steps, scores

def generate_plot(data, modes_to_plot, plot_raw_runs, out_path, spearman_cats, pearson_cats, colors, baseline_mode):
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])
    
    title = "Layer 5 RSR Training & Evaluation Trajectories (Mean ± StdErr)"
    if plot_raw_runs:
        title = "Layer 5 RSR Training & Evaluation Trajectories (Raw Runs)"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Plot Spearman Row
    axs_spearman = []
    
    for j, cat in enumerate(spearman_cats):
        ax = fig.add_subplot(gs[0, j])
        axs_spearman.append(ax)
        
        # Zero-shot baseline (Step 0 score)
        if baseline_mode in data and len(data[baseline_mode]["steps"]) > 0:
            baseline = data[baseline_mode]["scores"][cat]["mean"][0]
            ax.axhline(y=baseline, color='#4B5563', linestyle='--', alpha=0.8, label="Baseline")
            
        for mode in modes_to_plot:
            if mode in data:
                steps = data[mode]["steps"]
                mean = np.array(data[mode]["scores"][cat]["mean"], dtype=np.float64)
                stderr = np.array(data[mode]["scores"][cat]["stderr"], dtype=np.float64)
                raw = data[mode]["scores"][cat]["raw"]
                
                ax.plot(steps, mean, marker='o', markersize=4, color=colors[mode], label=mode, linewidth=1.8, zorder=3)
                
                if plot_raw_runs:
                    for i in range(len(raw)):
                        ax.plot(steps, raw[i], color=colors[mode], alpha=0.2, linewidth=1.0, zorder=2)
                else:
                    ax.fill_between(steps, mean - stderr, mean + stderr, color=colors[mode], alpha=0.15, zorder=2)
                
        ax.set_title(f"{cat}\n(Spearman rho)", fontsize=12, fontweight='semibold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("rho", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        if j == 0:
            ax.legend(loc='best', fontsize=9)
            
    # 2. Plot Pearson Row
    axs_pearson = []
    for j, cat in enumerate(pearson_cats):
        ax = fig.add_subplot(gs[1, j])
        axs_pearson.append(ax)
        
        # Zero-shot baseline (Step 0 score)
        if baseline_mode in data and len(data[baseline_mode]["steps"]) > 0:
            baseline = data[baseline_mode]["scores"][cat]["mean"][0]
            ax.axhline(y=baseline, color='#4B5563', linestyle='--', alpha=0.8, label="Baseline")
            
        for mode in modes_to_plot:
            if mode in data:
                steps = data[mode]["steps"]
                mean = np.array(data[mode]["scores"][cat]["mean"], dtype=np.float64)
                stderr = np.array(data[mode]["scores"][cat]["stderr"], dtype=np.float64)
                raw = data[mode]["scores"][cat]["raw"]
                
                ax.plot(steps, mean, marker='s', markersize=4, color=colors[mode], label=mode, linewidth=1.8, zorder=3)
                
                if plot_raw_runs:
                    for i in range(len(raw)):
                        ax.plot(steps, raw[i], color=colors[mode], alpha=0.2, linewidth=1.0, zorder=2)
                else:
                    ax.fill_between(steps, mean - stderr, mean + stderr, color=colors[mode], alpha=0.15, zorder=2)
                
        ax.set_title(f"{cat.replace(' (Pearson)', '')}\n(Pearson r)", fontsize=12, fontweight='semibold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("r", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        if j == 0:
            ax.legend(loc='best', fontsize=9)
            
    # Apply uniform y-axis scale to all 8 correlation plots
    all_corr_axs = axs_spearman + axs_pearson
    y_min, y_max = float('inf'), float('-inf')
    for ax in all_corr_axs:
        ymin, ymax = ax.get_ylim()
        if ymin < y_min: y_min = ymin
        if ymax > y_max: y_max = ymax
    for ax in all_corr_axs:
        ax.set_ylim(y_min, y_max)
            
    # 3. Plot MLM Training Loss Row (Spans all columns)
    ax_loss = fig.add_subplot(gs[2, :])
    
    for mode in modes_to_plot:
        if mode in data:
            steps = data[mode]["steps"]
            mean = np.array(data[mode]["scores"]["MLM Loss"]["mean"], dtype=np.float64)
            stderr = np.array(data[mode]["scores"]["MLM Loss"]["stderr"], dtype=np.float64)
            raw = np.array(data[mode]["scores"]["MLM Loss"]["raw"], dtype=np.float64)
            
            # Filter out step 0 since loss is N/A there
            valid_mask = ~np.isnan(mean)
            if np.any(valid_mask):
                valid_steps = np.array(steps)[valid_mask]
                ax_loss.plot(valid_steps, mean[valid_mask], marker='^', markersize=5, 
                        color=colors[mode], label=f"{mode} (MLM Training Loss)", linewidth=2.0, zorder=3)
                        
                if plot_raw_runs:
                    for i in range(raw.shape[0]):
                        ax_loss.plot(valid_steps, raw[i][valid_mask], color=colors[mode], alpha=0.2, linewidth=1.0, zorder=2)
                else:
                    ax_loss.fill_between(valid_steps, mean[valid_mask] - stderr[valid_mask], 
                                    mean[valid_mask] + stderr[valid_mask], color=colors[mode], alpha=0.15, zorder=2)
                
    ax_loss.set_title("MLM Training Loss Curve", fontsize=13, fontweight='bold')
    ax_loss.set_xlabel("Training Steps", fontsize=11)
    ax_loss.set_ylabel("Loss (Cross Entropy)", fontsize=11)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved successfully to {out_path}")

def generate_1x3_individual_plot(data, out_path):
    modes_to_plot = ["vanilla_all", "rsr_all"]
    categories = ["All pairs", "Both in RSR", "Neither in RSR"]
    colors = {
        "vanilla_all": "#3B82F6",
        "rsr_all": "#EF4444"
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Spearman Correlation Individual Replicates (vanilla_all vs rsr_all)", fontsize=15, fontweight='bold', y=0.98)
    
    baseline_mode = "vanilla_all"
    
    for j, cat in enumerate(categories):
        ax = axs[j]
        
        # Plot baseline (Step 0 score)
        if baseline_mode in data and len(data[baseline_mode]["steps"]) > 0:
            baseline = data[baseline_mode]["scores"][cat]["mean"][0]
            ax.axhline(y=baseline, color='#4B5563', linestyle='--', alpha=0.8, label="Baseline")
            
        for mode in modes_to_plot:
            if mode in data:
                steps = data[mode]["steps"]
                mean = data[mode]["scores"][cat]["mean"]
                raw = data[mode]["scores"][cat]["raw"]
                
                # Plot individual replicates
                for r_idx in range(len(raw)):
                    ax.plot(steps, raw[r_idx], color=colors[mode], alpha=0.15, linewidth=0.8, zorder=2)
                
                # Plot mean trajectory
                ax.plot(steps, mean, color=colors[mode], linewidth=2.5, label=f"{mode} (Mean)", zorder=3)
                
        ax.set_title(f"{cat} (Spearman rho)", fontsize=11, fontweight='semibold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("rho", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if j == 0:
            ax.legend(loc='best', fontsize=9)
            
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"1x3 Individual plot saved successfully to {out_path}")

def generate_1x3_all_modes_stderr_plot(data, out_path):
    modes = ["vanilla_all", "vanilla_targetwords", "rsr_all", "rsr_targetwords"]
    categories = ["All pairs", "Both in RSR", "Neither in RSR"]
    colors = {
        "vanilla_all": "#3B82F6",
        "vanilla_targetwords": "#10B981",
        "rsr_all": "#EF4444",
        "rsr_targetwords": "#F59E0B"
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Layer 5 RSR Spearman Correlation Trajectories (Mean ± StdErr)", fontsize=15, fontweight='bold', y=0.98)
    
    baseline_mode = "vanilla_all"
    
    for j, cat in enumerate(categories):
        ax = axs[j]
        
        # Plot baseline (Step 0 score)
        if baseline_mode in data and len(data[baseline_mode]["steps"]) > 0:
            baseline = data[baseline_mode]["scores"][cat]["mean"][0]
            ax.axhline(y=baseline, color='#4B5563', linestyle='--', alpha=0.8, label="Baseline")
            
        for mode in modes:
            if mode in data:
                steps = data[mode]["steps"]
                mean = np.array(data[mode]["scores"][cat]["mean"], dtype=np.float64)
                stderr = np.array(data[mode]["scores"][cat]["stderr"], dtype=np.float64)
                
                ax.plot(steps, mean, color=colors[mode], linewidth=2.0, label=mode, marker='o', markersize=3, zorder=3)
                ax.fill_between(steps, mean - stderr, mean + stderr, color=colors[mode], alpha=0.15, zorder=2)
                
        ax.set_title(f"{cat} (Spearman rho)", fontsize=11, fontweight='semibold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("rho", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if j == 0:
            ax.legend(loc='best', fontsize=9)
            
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"1x3 Stderr plot saved successfully to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot RSR Training Trajectories")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory where evaluation logs are saved.")
    args = parser.parse_args()
    
    modes = [
        "vanilla_all",
        "vanilla_targetwords",
        "rsr_all",
        "rsr_targetwords"
    ]
    
    categories = [
        "MLM Loss",
        "All pairs",
        "Both in RSR",
        "One in RSR",
        "Neither in RSR",
        "All pairs (Pearson)",
        "Both in RSR (Pearson)",
        "One in RSR (Pearson)",
        "Neither in RSR (Pearson)"
    ]
    
    data = {}
    
    # Data Aggregation
    for mode in modes:
        mode_scores = {cat: [] for cat in categories}
        mode_steps = None
        
        # Look for replicates
        for run_id in range(10):
            filepath = os.path.join(args.results_dir, f"eval_log_{mode}_run{run_id}.txt")
            if os.path.exists(filepath):
                steps, scores = parse_log_file(filepath)
                if mode_steps is None:
                    mode_steps = steps
                
                # Only include runs that match expected step lengths
                if len(steps) == len(mode_steps):
                    for cat in categories:
                        # Clean None values into np.nan
                        cleaned = [x if x is not None else np.nan for x in scores[cat]]
                        mode_scores[cat].append(cleaned)
                else:
                    print(f"Skipping incomplete run: {filepath} (len={len(steps)}, expected={len(mode_steps)})")
            else:
                pass
                
        if mode_steps is not None and len(mode_scores["All pairs"]) > 0:
            aggregated_scores = {}
            for cat in categories:
                arr = np.array(mode_scores[cat], dtype=np.float32) # Shape: (num_runs, num_steps)
                # Compute valid mask to ignore nans properly
                valid_counts = np.sum(~np.isnan(arr), axis=0)
                # Suppress runtime warnings for empty slices (all NaNs)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean = np.nanmean(arr, axis=0)
                    std = np.nanstd(arr, axis=0)
                    stderr = np.where(valid_counts > 0, std / np.sqrt(valid_counts + 1e-8), np.nan)
                    
                # Convert back to regular python floats/lists to serialize to JSON later
                aggregated_scores[cat] = {
                    "mean": [float(x) if not np.isnan(x) else None for x in mean],
                    "stderr": [float(x) if not np.isnan(x) else None for x in stderr],
                    "raw": [[float(x) if not np.isnan(x) else None for x in run] for run in arr]
                }
            
            data[mode] = {"steps": mode_steps, "scores": aggregated_scores}
            
    if not data:
        print("No completed log files found. Cannot generate plot.")
        return
        
    # Dump results to CSV
    csv_rows = []
    for mode, mode_data in data.items():
        steps = mode_data["steps"]
        for i, step in enumerate(steps):
            row = {"Mode": mode, "Step": step}
            for cat in categories:
                row[f"{cat} (Mean)"] = mode_data["scores"][cat]["mean"][i]
                row[f"{cat} (StdErr)"] = mode_data["scores"][cat]["stderr"][i]
            csv_rows.append(row)
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(args.results_dir, "aggregated_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Aggregated means saved to {csv_path}")
        
    # Dump complete structured dictionary to JSON for future re-plotting
    json_path = os.path.join(args.results_dir, "all_trajectories_data.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Full trajectory data (including raw runs) saved to {json_path}")
        
    colors = {
        "vanilla_all": "#3B82F6",         # Royal Blue
        "vanilla_targetwords": "#10B981", # Emerald Green
        "rsr_all": "#EF4444",             # Coral Red
        "rsr_targetwords": "#F59E0B"      # Amber Gold
    }
    
    spearman_cats = ["All pairs", "Both in RSR", "One in RSR", "Neither in RSR"]
    pearson_cats = ["All pairs (Pearson)", "Both in RSR (Pearson)", "One in RSR (Pearson)", "Neither in RSR (Pearson)"]
    
    os.makedirs(args.results_dir, exist_ok=True)
    baseline_mode = list(data.keys())[0] if data else None
    
    # Generate the 6 permutations (including 3x4 full reports and 1x3 clean reports)
    
    # 1. 1x3 All Modes (StdErr) - Spearman only
    generate_1x3_all_modes_stderr_plot(data, os.path.join(args.results_dir, "all_modes_stderr.png"))
                  
    # 2. 1x3 Vanilla & RSR All (Individual Runs) - Spearman only
    generate_1x3_individual_plot(data, os.path.join(args.results_dir, "vanilla_rsr_all_individual.png"))
                  
    # 3. 3x4 Full All Modes (StdErr) - Spearman, Pearson, and MLM Loss
    generate_plot(data, modes, plot_raw_runs=False, 
                  out_path=os.path.join(args.results_dir, "all_modes_full_stderr.png"), 
                  spearman_cats=spearman_cats, pearson_cats=pearson_cats, 
                  colors=colors, baseline_mode=baseline_mode)
                  
    # 4. 3x4 Full All Modes (Individual Runs) - Spearman, Pearson, and MLM Loss
    generate_plot(data, modes, plot_raw_runs=True, 
                  out_path=os.path.join(args.results_dir, "all_modes_individual.png"), 
                  spearman_cats=spearman_cats, pearson_cats=pearson_cats, 
                  colors=colors, baseline_mode=baseline_mode)
                  
    # 5. 3x4 Full Vanilla & RSR All (StdErr) - Spearman, Pearson, and MLM Loss
    restricted_modes = ["vanilla_all", "rsr_all"]
    generate_plot(data, restricted_modes, plot_raw_runs=False, 
                  out_path=os.path.join(args.results_dir, "vanilla_rsr_all_stderr.png"), 
                  spearman_cats=spearman_cats, pearson_cats=pearson_cats, 
                  colors=colors, baseline_mode=baseline_mode)
                  
    # 6. 3x4 Full Vanilla & RSR All (Individual Runs) - Spearman, Pearson, and MLM Loss
    generate_plot(data, restricted_modes, plot_raw_runs=True, 
                  out_path=os.path.join(args.results_dir, "vanilla_rsr_all_full_individual.png"), 
                  spearman_cats=spearman_cats, pearson_cats=pearson_cats, 
                  colors=colors, baseline_mode=baseline_mode)

if __name__ == "__main__":
    main()
