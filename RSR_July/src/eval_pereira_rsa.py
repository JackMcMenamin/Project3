"""
Evaluation script for testing models against the Pereira (2018) fMRI concept dataset.

Workflow Context:
This script forms part of the downstream biological evaluation pipeline for the Representational 
Similarity Regularisation (RSR) project. After models (e.g., `vanilla_all` and `rsr_all`) have been 
fine-tuned on Wikipedia, this script tests if the geometric alignment enforced by RSR during training 
provides a better zero-shot fit to human neural representations.

It performs two primary evaluations:
1. Zero-shot Representation Similarity Analysis (RSA): Correlating the artificial RDM with the fMRI RDM.
2. 2vs2 Match testing: A pairwise discriminability test.

This analysis is conducted independently across all available subjects (e.g., 10 subjects) using robust `wiki_avg` 
contextual pooling to extract prototypical vector representations from deep layers.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, sem
from transformers import RobertaTokenizer, RobertaForMaskedLM
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import find_subword_span

def load_words(filepath):
    with open(filepath, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    return words

def extract_layer_representations(model, tokenizer, words, device, mode, wiki_sentences=None):
    """
    Extracts deep layer representations for a list of target concepts.
    If mode == 'wiki_avg', it extracts the representations of the concepts within 
    natural wikipedia contexts (up to 20 sentences) and averages them to form stable prototypes.
    Uses subword span matching to robustly extract token representations.
    """
    model.eval()
    layer_reps = {i: [] for i in range(13)}
    
    with torch.no_grad():
        for word in words:
            sents = []
            if mode == 'bare':
                sents = [word]
            elif mode == 'wiki_avg':
                available = wiki_sentences.get(word, [])
                if len(available) >= 30:
                    eval_pool = available[30:]
                else:
                    eval_pool = available
                
                for s in eval_pool:
                    if len(tokenizer.encode(s, add_special_tokens=True)) <= 512:
                        sents.append(s)
                    if len(sents) == 20:
                        break
                if not sents:
                    sents = [word]
            
            word_layer_vectors = {i: [] for i in range(13)}
            
            for s in sents:
                span = find_subword_span(tokenizer, s, word)
                if span is None:
                    span = (1, len(tokenizer.encode(s)) - 1)
                    
                inputs = tokenizer(s, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                for i, h in enumerate(hidden_states):
                    start_idx, end_idx = span
                    if start_idx >= h.size(1) or end_idx > h.size(1):
                        rep = h[0, 1:-1, :].mean(dim=0).cpu().numpy()
                    else:
                        span_hidden = h[0, start_idx:end_idx, :]
                        if span_hidden.size(0) == 0:
                            rep = h[0, 1:-1, :].mean(dim=0).cpu().numpy()
                        else:
                            rep = span_hidden.mean(dim=0).cpu().numpy()
                    word_layer_vectors[i].append(rep)
                    
            for i in range(13):
                layer_reps[i].append(np.mean(word_layer_vectors[i], axis=0))
                
    for i in range(13):
        layer_reps[i] = np.array(layer_reps[i])
        
    return layer_reps

def compute_similarity_matrix(reps):
    num_words = reps.shape[0]
    sim_matrix = np.zeros((num_words, num_words))
    for i in range(num_words):
        for j in range(num_words):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(reps[i], reps[j])
                sim_matrix[i, j] = corr
    return sim_matrix

def test_2vs2(M_D, M_B):
    """
    Computes the 2vs2 Match Test Accuracy.
    Evaluates whether the geometric distance between two concepts in the model's representation (M_D)
    matches the distance in the brain's representation (M_B) better than a mismatched pair.
    """
    N = M_D.shape[0]
    
    # Precompute sums and sum of squares for each row
    sum_D = M_D.sum(axis=1)
    sum_B = M_B.sum(axis=1)
    sum_D2 = (M_D**2).sum(axis=1)
    sum_B2 = (M_B**2).sum(axis=1)
    
    dot_DB = M_D @ M_B.T  # Shape: (N, N)
    
    correct = 0
    total = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            def get_r(sum_X_full, sum_Y_full, sum_X2_full, sum_Y2_full, dot_XY_full, val_X_i, val_X_j, val_Y_i, val_Y_j):
                sX = sum_X_full - val_X_i - val_X_j
                sY = sum_Y_full - val_Y_i - val_Y_j
                sX2 = sum_X2_full - val_X_i**2 - val_X_j**2
                sY2 = sum_Y2_full - val_Y_i**2 - val_Y_j**2
                sXY = dot_XY_full - val_X_i * val_Y_i - val_X_j * val_Y_j
                
                n = N - 2
                meanX = sX / n
                meanY = sY / n
                varX = sX2 / n - meanX**2
                varY = sY2 / n - meanY**2
                
                cov = sXY / n - meanX * meanY
                
                stdX = np.sqrt(max(0, varX))
                stdY = np.sqrt(max(0, varY))
                
                if stdX == 0 or stdY == 0:
                    return 0.0
                return cov / (stdX * stdY)
            
            r_Di_Bi = get_r(sum_D[i], sum_B[i], sum_D2[i], sum_B2[i], dot_DB[i, i], M_D[i, i], M_D[i, j], M_B[i, i], M_B[i, j])
            r_Dj_Bj = get_r(sum_D[j], sum_B[j], sum_D2[j], sum_B2[j], dot_DB[j, j], M_D[j, i], M_D[j, j], M_B[j, i], M_B[j, j])
            r_Di_Bj = get_r(sum_D[i], sum_B[j], sum_D2[i], sum_B2[j], dot_DB[i, j], M_D[i, i], M_D[i, j], M_B[j, i], M_B[j, j])
            r_Dj_Bi = get_r(sum_D[j], sum_B[i], sum_D2[j], sum_B2[i], dot_DB[j, i], M_D[j, i], M_D[j, j], M_B[i, i], M_B[i, j])
            
            if (r_Di_Bi + r_Dj_Bj) > (r_Di_Bj + r_Dj_Bi):
                correct += 1
            total += 1
            
    return correct / total

def compute_rsa(M_D, M_B, mask=None):
    N = M_D.shape[0]
    triu_indices = np.triu_indices(N, k=1)
    
    if mask is not None:
        triu_mask = np.zeros((N, N), dtype=bool)
        triu_mask[triu_indices] = True
        final_mask = triu_mask & mask
        D_vec = M_D[final_mask]
        B_vec = M_B[final_mask]
    else:
        D_vec = M_D[triu_indices]
        B_vec = M_B[triu_indices]
        
    if len(D_vec) > 1:
        corr, _ = spearmanr(D_vec, B_vec)
        return corr
    return 0.0

def load_or_simulate_brain_matrix(filepath, words):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df_mean = df.groupby(['concept', 'voxel'])['beta'].mean().reset_index()
        df_pivot = df_mean.pivot(index='concept', columns='voxel', values='beta')
        df_pivot.index = df_pivot.index.str.lower()
        words_lower = [w.lower() for w in words]
        df_pivot = df_pivot.reindex(words_lower)
        if df_pivot.isna().any().any():
            df_pivot = df_pivot.fillna(0)
        emp_mat = df_pivot.to_numpy()
        return compute_similarity_matrix(emp_mat)
    else:
        rand_mat = np.random.randn(len(words), 1000)
        return compute_similarity_matrix(rand_mat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results_downstream_bdhomepc_2026-07-05", help="Directory to save the output plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    words_path = "data/pereira_2018/Pereira_Materials/stimuli_180concepts.txt"
    words = load_words(words_path)
    print(f"Loaded {len(words)} concepts.")
    
    rsr_words = set()
    try:
        with open("data/wiki_targetwords.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                rsr_words.add(data['word'].lower())
    except Exception as e:
        print(f"WARNING: Could not load wiki_targetwords.jsonl: {e}")
        
    N = len(words)
    in_rsr = np.array([w.lower() in rsr_words for w in words], dtype=bool)
    rsr_count_matrix = in_rsr[:, None].astype(int) + in_rsr[None, :].astype(int)
    
    masks = {
        "All pairs": None,
        "Both in RSR": (rsr_count_matrix == 2),
        "One in RSR": (rsr_count_matrix == 1),
        "Neither in RSR": (rsr_count_matrix == 0)
    }
    
    wiki_sentences = {}
    wiki_path = "data/wiki_pereira.jsonl"
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                wiki_sentences[data['word']] = data['sentences']
                
    # Multi-subject loading
    outputs_dir = "data/ryskina_repo/outputs/rsa"
    subject_files = [f for f in os.listdir(outputs_dir) if f.startswith("betas_sentences_") and f.endswith(".csv")]
    subjects = sorted([f.split('_')[-1].replace('.csv', '') for f in subject_files])
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    M_B_dict = {}
    for subj in subjects:
        M_B_dict[subj] = load_or_simulate_brain_matrix(os.path.join(outputs_dir, f"betas_sentences_{subj}.csv"), words)
    
    model_paths = {
        "roberta-base": "roberta-base",
        "vanilla_all": "results/vanilla_all_checkpoint_8000_run1.pt",
        "rsr_all": "results/rsr_all_checkpoint_8000_run1.pt"
    }
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    modes = ['bare', 'wiki_avg']
    # results format: results[mode][model_name][layer][subject]
    results_2v2 = {mode: {name: {L: [] for L in range(13)} for name in model_paths.keys()} for mode in modes}
    results_rsa = {mode: {name: {cat: {L: [] for L in range(13)} for cat in masks.keys()} for name in model_paths.keys()} for mode in modes}
    layer5_MDs = {mode: {} for mode in modes}
    
    for model_name, path in model_paths.items():
        print(f"Loading {model_name}...")
        if model_name == "roberta-base":
            model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        else:
            model = RobertaForMaskedLM.from_pretrained('roberta-base')
            state_dict = torch.load(path, map_location='cpu')
            new_state_dict = {k[len('roberta_mlm.'):] if k.startswith('roberta_mlm.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.to(device)
            
        for mode in modes:
            print(f"  Evaluating {model_name} in {mode} mode...")
            layer_reps = extract_layer_representations(model, tokenizer, words, device, mode, wiki_sentences)
            
            for layer in range(13):
                M_D = compute_similarity_matrix(layer_reps[layer])
                if layer == 5:
                    layer5_MDs[mode][model_name] = M_D
                
                for subj in subjects:
                    M_B = M_B_dict[subj]
                    acc_2v2 = test_2vs2(M_D, M_B)
                    results_2v2[mode][model_name][layer].append(acc_2v2)
                    
                    for cat, mask in masks.items():
                        rsa_val = compute_rsa(M_D, M_B, mask)
                        results_rsa[mode][model_name][cat][layer].append(rsa_val)
                        
    # Plotting Functions
    def plot_with_error_bands(ax, data_dict, modes, models, colors, ylabel, title):
        for name in models:
            for mode in modes:
                means = []
                sems = []
                for L in range(13):
                    scores = data_dict[mode][name][L]
                    means.append(np.mean(scores))
                    sems.append(sem(scores) if len(scores) > 1 else 0.0)
                
                means = np.array(means)
                sems = np.array(sems)
                
                ls = '--' if mode == 'bare' else '-'
                marker = '' if mode == 'bare' else 'o'
                alpha = 0.5 if mode == 'bare' else 1.0
                
                ax.plot(range(13), means, color=colors[name], linestyle=ls, marker=marker, alpha=alpha, label=f"{name} ({mode})")
                ax.fill_between(range(13), means - sems, means + sems, color=colors[name], alpha=0.15 if mode == 'wiki_avg' else 0.05)
                
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    colors = {"roberta-base": "blue", "vanilla_all": "orange", "rsr_all": "green"}
    
    # 2v2 Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_error_bands(ax, results_2v2, modes, model_paths.keys(), colors, "Accuracy", "2 vs 2 Matching Accuracy (Pereira 2018 Concepts)")
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance (0.5)')
    out_2v2 = os.path.join(args.output_dir, f"pereira_2v2_accuracy_{len(subjects)}subjs.png")
    plt.savefig(out_2v2)
    
    # RSA Plot (All pairs)
    fig, ax = plt.subplots(figsize=(10, 6))
    rsa_all_pairs = {m: {name: {L: results_rsa[m][name]['All pairs'][L] for L in range(13)} for name in model_paths.keys()} for m in modes}
    plot_with_error_bands(ax, rsa_all_pairs, modes, model_paths.keys(), colors, "Spearman Correlation", "Representation Similarity Analysis (Pereira 2018)")
    out_rsa = os.path.join(args.output_dir, f"pereira_rsa_{len(subjects)}subjs.png")
    plt.savefig(out_rsa)
    
    # Categorised RSA Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, cat in enumerate(masks.keys()):
        rsa_cat = {m: {name: {L: results_rsa[m][name][cat][L] for L in range(13)} for name in model_paths.keys()} for m in modes}
        plot_with_error_bands(axes[i], rsa_cat, modes, model_paths.keys(), colors, "Spearman r", f"RSA: {cat}")
    plt.tight_layout()
    out_rsa_cat = os.path.join(args.output_dir, f"pereira_rsa_categorized_{len(subjects)}subjs.png")
    plt.savefig(out_rsa_cat)
    
    # Plot RDMs for Layer 5 (Using wiki_avg against average brain RDM)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    avg_M_B = np.mean(list(M_B_dict.values()), axis=0)
    im0 = axes[0].imshow(1 - avg_M_B, cmap='viridis', vmin=0, vmax=2)
    axes[0].set_title("Empirical Brain Data (Avg RDM)")
    fig.colorbar(im0, ax=axes[0])
    
    for idx, name in enumerate(model_paths.keys()):
        im = axes[idx+1].imshow(1 - layer5_MDs['wiki_avg'][name], cmap='viridis', vmin=0, vmax=2)
        axes[idx+1].set_title(f"{name} Layer 5 (RDM) [wiki_avg]")
        fig.colorbar(im, ax=axes[idx+1])
        
    plt.tight_layout()
    out_rdm = os.path.join(args.output_dir, f"pereira_rdm_layer5_{len(subjects)}subjs.png")
    plt.savefig(out_rdm)
    
    print(f"\nPlots saved to {out_2v2}, {out_rsa}, and {out_rdm}")

    out_json_2v2 = os.path.join(args.output_dir, f"pereira_2v2_results_{len(subjects)}subjs.json")
    out_json_rsa = os.path.join(args.output_dir, f"pereira_rsa_results_{len(subjects)}subjs.json")
    with open(out_json_2v2, "w") as f:
        json.dump(results_2v2, f, indent=4)
    with open(out_json_rsa, "w") as f:
        json.dump(results_rsa, f, indent=4)
        
if __name__ == "__main__":
    main()
