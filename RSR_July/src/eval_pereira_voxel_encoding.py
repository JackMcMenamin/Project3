"""
Voxelwise Encoding Evaluation Script (Ridge Regression)

Workflow Context:
This script performs out-of-fold generalization testing as part of the biological evaluation 
pipeline for the RSR project. It trains an independent RidgeCV regression model (with 10-fold CV) 
to predict the activation of individual language-sensitive fMRI voxels (2,296 language-SN220 parcels) 
based on the 768-dimensional language model representations.

By predicting actual brain activation directly from the model representations out-of-fold, 
it ensures that the geometry captures genuine structural alignment rather than global correlational artifacts.
This is computed independently across all available subjects (e.g., 10 subjects) from the Pereira (2018) fMRI dataset.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, sem
from transformers import RobertaTokenizer, RobertaForMaskedLM
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import find_subword_span
from eval_pereira_rsa import load_words, extract_layer_representations

def load_raw_brain_matrix(filepath, words):
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
        return emp_mat
    else:
        rand_mat = np.random.randn(len(words), 1000)
        return rand_mat

def evaluate_encoding(X, Y, n_splits=10):
    """
    Evaluates the voxelwise encoding performance using a cross-validated Ridge Regression.
    Trains on 90% of the concepts to predict the voxel activations (Y) using the 
    model's representations (X). Tests on the remaining 10% (out-of-fold generalization)
    and computes the mean Pearson correlation across all language voxels.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    alphas = np.logspace(-2, 4, 7)
    
    fold_stimulus_r = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model = RidgeCV(alphas=alphas)
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        
        voxel_r = []
        for v in range(Y_test.shape[1]):
            r, _ = pearsonr(Y_test[:, v], preds[:, v])
            if np.isnan(r):
                r = 0.0
            voxel_r.append(r)
            
        fold_stimulus_r.append(np.mean(voxel_r))
        
    return np.mean(fold_stimulus_r)

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
    
    wiki_sentences = {}
    wiki_path = "data/wiki_pereira.jsonl"
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                wiki_sentences[data['word']] = data['sentences']
                
    outputs_dir = "data/ryskina_repo/outputs/rsa"
    subject_files = [f for f in os.listdir(outputs_dir) if f.startswith("betas_sentences_") and f.endswith(".csv")]
    subjects = sorted([f.split('_')[-1].replace('.csv', '') for f in subject_files])
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    Y_brain_dict = {}
    for subj in subjects:
        Y_brain_dict[subj] = load_raw_brain_matrix(os.path.join(outputs_dir, f"betas_sentences_{subj}.csv"), words)
    
    model_paths = {
        "roberta-base": "roberta-base",
        "vanilla_all": "results/vanilla_all_checkpoint_8000_run1.pt",
        "rsr_all": "results/rsr_all_checkpoint_8000_run1.pt"
    }
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # results format: results[model_name][layer][subject]
    results_encoding = {name: {L: [] for L in range(13)} for name in model_paths.keys()}
    
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
            
        print(f"  Evaluating {model_name} encoding models...")
        layer_reps = extract_layer_representations(model, tokenizer, words, device, mode='wiki_avg', wiki_sentences=wiki_sentences)
        
        for layer in range(13):
            X_layer = layer_reps[layer]
            
            for subj in subjects:
                Y_brain = Y_brain_dict[subj]
                score = evaluate_encoding(X_layer, Y_brain, n_splits=10)
                results_encoding[model_name][layer].append(score)
                
            mean_score = np.mean(results_encoding[model_name][layer])
            print(f"    Layer {layer}: Mean Pearson r across {len(subjects)} subjects = {mean_score:.4f}")

    # Plot Encoding Accuracy
    plt.figure(figsize=(10, 6))
    colors = {"roberta-base": "blue", "vanilla_all": "orange", "rsr_all": "green"}
    
    for name in model_paths.keys():
        means = []
        sems = []
        for L in range(13):
            scores = results_encoding[name][L]
            means.append(np.mean(scores))
            sems.append(sem(scores) if len(scores) > 1 else 0.0)
            
        means = np.array(means)
        sems = np.array(sems)
        
        plt.plot(range(13), means, color=colors[name], marker='o', label=name)
        plt.fill_between(range(13), means - sems, means + sems, color=colors[name], alpha=0.15)
        
    plt.title("Voxelwise Encoding Score (Ridge Regression 10-Fold CV)")
    plt.xlabel("Layer")
    plt.ylabel("Mean Voxelwise Pearson r (± SEM)")
    plt.legend()
    plt.grid(True)
    out_plot = os.path.join(args.output_dir, f"pereira_encoding_layerwise_{len(subjects)}subjs.png")
    plt.savefig(out_plot)
    print(f"\nPlot saved to {out_plot}")
    
    # Dump raw json results
    out_json = os.path.join(args.output_dir, f"pereira_encoding_results_{len(subjects)}subjs.json")
    with open(out_json, "w") as f:
        json.dump(results_encoding, f, indent=4)

if __name__ == "__main__":
    main()
