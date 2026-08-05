"""
RoBERTa Layer Evaluation Script

This script evaluates the baseline semantic representations of an untrained 
RoBERTa model across all of its layers (0 to 12). It compares the representational 
geometry of words presented in isolation ('bare' mode) versus words presented in 
naturally occurring sentence contexts ('wiki_avg' mode).

The outputs are the Spearman rank correlations against the SimLex-999 dataset,
plotted and saved as a graph for analysis.

Workflow Context:
- A standalone utility script used to determine which layer of RoBERTa contains 
  the most robust semantic representations prior to any RSR fine-tuning.
"""

import torch
import torch.nn.functional as F
import scipy.stats as stats
from transformers import RobertaForMaskedLM, RobertaTokenizerFast as RobertaTokenizer
import json
import os
import matplotlib.pyplot as plt
import numpy as np



def main():
    """
    Main execution routine for layer-wise evaluation.
    
    Process:
    1. Loads the base RoBERTa model, tokenizer, and SimLex-999 evaluation dataset.
    2. Builds a dictionary of Wikipedia sentence contexts for each target word.
    3. Runs a forward pass to extract hidden states from all 13 layers (Embedding + 12 Transformer layers).
    4. Computes the cosine similarity of word pairs and calculates Spearman correlation for each layer.
    5. Plots and saves a comparative performance graph.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A GPU is required to run the evaluation.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    print("Loading Roberta-base...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
    model.eval()

    print("Loading SimLex-999 and Training Data for categorisation...")
    from data_prep import load_simlex999, find_subword_span, load_and_standardize_datasets
    training_data = load_and_standardize_datasets()
    simlex_pairs = load_simlex999()
    
    # Extract words that the model *would* see during RSR training
    training_words = set()
    for w1, w2, _ in training_data:
        training_words.add(w1)
        training_words.add(w2)

    wiki_sentences_dict = {}
    target_path = os.path.join("data", "wiki_targetwords.jsonl")
    if os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                filtered_sentences = [s for s in data['sentences'] if len(s.split()) <= 100]
                # Use only the evaluation half (second half) of the sentence pool
                midpoint = len(filtered_sentences) // 2
                wiki_sentences_dict[data['word']] = filtered_sentences[midpoint:]

    unique_words = set()
    for w1, w2, _ in simlex_pairs:
        unique_words.add(w1)
        unique_words.add(w2)

    word_vectors = {"bare": {}, "wiki_avg": {}}

    for mode in ["bare", "wiki_avg"]:
        print(f"Computing representations for mode: {mode}")
        batch_size = 64 if mode == "bare" else 32
        unique_words_list = list(unique_words)
        
        for i in range(0, len(unique_words_list), batch_size):
            batch_words = unique_words_list[i:i+batch_size]
            
            all_sents = []
            all_spans = []
            word_sent_counts = []
            
            for w in batch_words:
                if mode == "wiki_avg":
                    sents = []
                    for s in wiki_sentences_dict.get(w, []):
                        if len(tokenizer.encode(s, add_special_tokens=True)) <= 512:
                            sents.append(s)
                        if len(sents) == 20:
                            break
                    if not sents:
                        raise ValueError(f"wiki_avg mode: no valid wiki sentences found for word '{w}'. Ensure wiki_targetwords.jsonl is populated.")
                else:
                    sents = [w]
                    
                all_sents.extend(sents)
                word_sent_counts.append(len(sents))
                for s in sents:
                    span = find_subword_span(tokenizer, s, w)
                    if span is None:
                        raise ValueError(f"find_subword_span failed to locate '{w}' in sentence: '{s}'")
                    all_spans.append(span)
                    
            # Tokenize the full batch of sentences
            encoded = tokenizer(all_sents, padding=True, return_tensors="pt").to(device)
            
            # Forward pass without gradients to extract all hidden states
            with torch.no_grad():
                outputs = model.roberta(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], output_hidden_states=True)
                
            # Iterate through all 13 layers (0 = Embedding, 1-12 = Transformer layers)
            for layer_idx in range(13):
                hidden_states = outputs.hidden_states[layer_idx]
                
                idx = 0
                # Process each target word and pool its contexts
                for w, count in zip(batch_words, word_sent_counts):
                    pooled_vectors = []
                    for k in range(count):
                        start_idx, end_idx = all_spans[idx + k]
                        
                        if start_idx >= hidden_states.size(1) or end_idx > hidden_states.size(1):
                            raise ValueError(f"Span ({start_idx}, {end_idx}) exceeds hidden state length {hidden_states.size(1)}")
                            
                        span_hidden = hidden_states[idx + k, start_idx:end_idx, :]
                        if span_hidden.size(0) == 0:
                            raise ValueError(f"Empty span slice for indices ({start_idx}, {end_idx})")
                            
                        # Mean-pool the subword tokens for this specific context
                        pooled_vectors.append(torch.mean(span_hidden, dim=0))
                        
                    # Stack all contexts for the word and mean-pool them to get the final representation
                    pooled_vectors = torch.stack(pooled_vectors)
                    word_vec = pooled_vectors.mean(dim=0).cpu().numpy()
                    
                    if w not in word_vectors[mode]:
                        word_vectors[mode][w] = []
                    # Append the extracted vector for this layer
                    if len(word_vectors[mode][w]) <= layer_idx:
                        word_vectors[mode][w].append(word_vec)
                    else:
                        word_vectors[mode][w][layer_idx] = word_vec
                        
                    idx += count

    results = {"bare": [], "wiki_avg": []}
    
    print("\n--- Evaluation Results ---")
    for mode in ["bare", "wiki_avg"]:
        print(f"\nMode: {mode}")
        for layer in range(13):
            categories = {
                "All pairs": [],
                "Both in RSR": [],
                "One in RSR": [],
                "Neither in RSR": []
            }
            
            for w1, w2, human_score in simlex_pairs:
                v1 = word_vectors[mode][w1][layer]
                v2 = word_vectors[mode][w2][layer]
                # Cosine similarity
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                in1 = w1 in training_words
                in2 = w2 in training_words
                data_point = (cos_sim, human_score)
                
                categories["All pairs"].append(data_point)
                if in1 and in2:
                    categories["Both in RSR"].append(data_point)
                elif in1 or in2:
                    categories["One in RSR"].append(data_point)
                else:
                    categories["Neither in RSR"].append(data_point)
                
            layer_results = {}
            for cat, items in categories.items():
                if len(items) > 1:
                    preds = [p for p, t in items]
                    targets = [t for p, t in items]
                    rho, _ = stats.spearmanr(preds, targets)
                    layer_results[cat] = rho
                else:
                    layer_results[cat] = 0.0
                    
            # For the plot, we just track "All pairs"
            results[mode].append(layer_results["All pairs"])
            
            print(f"  Layer {layer:2d}: All={layer_results['All pairs']:.4f} | Both={layer_results['Both in RSR']:.4f} | One={layer_results['One in RSR']:.4f} | Neither={layer_results['Neither in RSR']:.4f}")

    # Plotting
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    layers = list(range(13))
    plt.plot(layers, results["bare"], marker='o', label="Bare ([CLS] word [SEP])")
    plt.plot(layers, results["wiki_avg"], marker='x', label="Wiki Avg (20 contexts)")
    plt.title("SimLex-999 Spearman Correlation by Untrained RoBERTa Layer")
    plt.xlabel("Layer (0 = Embedding, 1-12 = Transformer Layers)")
    plt.ylabel("Spearman Rho")
    plt.xticks(layers)
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join("results", "roberta_layers_eval.png")
    plt.savefig(save_path)
    print(f"\nSaved plot to {save_path}")

if __name__ == "__main__":
    main()
