"""
Evaluation Module for Representational Similarity Regularisation (RSR)

This module handles testing the trained model against the SimLex-999 dataset.
SimLex-999 provides rigorous, human-annotated similarity scores for 999 word pairs,
specifically distinguishing true semantic similarity from broader conceptual association.

The evaluation computes Spearman rank correlation across several subsets of the data
to determine how well the learned geometry generalizes.

Workflow Context:
- Consumed by: `train.py`. Called repeatedly during the main training loop (e.g., every 100 steps) 
  to log live performance trajectories.
- Inputs: Expects a `RobertaRSRModel` and the pre-loaded `simlex_pairs` dataset.
- Outputs: Returns a dictionary of Spearman correlations that are written directly into `eval_log.txt`.

Functions:
- evaluate_simlex: Evaluates the model on SimLex-999 and breaks down results by category.
"""

import torch
import torch.nn.functional as F
import scipy.stats as stats
from data_prep import find_subword_span

def evaluate_simlex(model, tokenizer, simlex_pairs, training_words, eval_mode="bare", wiki_sentences_dict=None, device="cpu"):
    """
    Evaluates the model's semantic representations against the SimLex-999 benchmark.
    
    This function processes the SimLex-999 dataset through the model to obtain 
    extracted semantic vectors. It computes cosine similarities between pairs and 
    then uses SciPy to calculate the exact Spearman rank correlation against 
    the human ratings.
    
    Crucially, it breaks down performance into four categories:
    - All pairs: Overall performance on all 999 items.
    - Both in RSR: Pairs where BOTH words were seen during training.
    - One in RSR: Pairs where exactly ONE word was seen during training.
    - Neither in RSR: Zero-shot performance on pairs where NEITHER word was seen.
    
    Args:
        model (RobertaRSRModel): The trained model wrapper.
        tokenizer (RobertaTokenizer): The HuggingFace tokenizer.
        simlex_pairs (list of tuple): List of (word1, word2, human_score) tuples.
        training_words (set of str): Words that were exposed to the model during training.
        eval_mode (str): Evaluation mode, 'bare' or 'wiki_avg'.
        wiki_sentences_dict (dict): Dictionary mapping words to lists of context sentences.
        device (torch.device): Compute device ('cpu' or 'cuda').
        
    Returns:
        dict: A dictionary mapping the four evaluation categories to their Spearman rho scores.
    """
    model.eval()
    
    # Categorization buckets for tracking correlations based on training exposure
    categories = {
        "All pairs": [],
        "Both in RSR": [],
        "One in RSR": [],
        "Neither in RSR": []
    }
    
    wiki_sentences_dict = wiki_sentences_dict or {}
    
    unique_words = list(set([w1 for w1, w2, _ in simlex_pairs] + [w2 for w1, w2, _ in simlex_pairs]))
    word_vectors = {}
    
    # Batch size of words. wiki_avg expands each word to 20 contexts, so we raise it for the high-VRAM GPU.
    batch_size = 64 if eval_mode == "bare" else 32 
    
    with torch.no_grad():
        for i in range(0, len(unique_words), batch_size):
            batch_words = unique_words[i:i+batch_size]
            
            all_sents = []
            all_spans = []
            word_sent_counts = []
            
            for w in batch_words:
                if eval_mode == "wiki_avg":
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
                    
            encoded = tokenizer(all_sents, padding=True, return_tensors="pt").to(device)
            vecs = model.target_vectors(encoded['input_ids'], encoded['attention_mask'], all_spans, apply_standardisation=False)
            
            idx = 0
            for w, count in zip(batch_words, word_sent_counts):
                word_vec = vecs[idx:idx+count].mean(dim=0)
                word_vectors[w] = word_vec
                idx += count

    with torch.no_grad():
        for w1, w2, human_score in simlex_pairs:
            v1 = word_vectors[w1]
            v2 = word_vectors[w2]
            
            cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            
            # Determine which category this pair belongs to based on training set exposure
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
                
    results = {}
    
    # Calculate both Spearman rho and Pearson r correlation for each populated category
    for cat, items in categories.items():
        if len(items) > 1:
            preds = [p for p, t in items]
            targets = [t for p, t in items]
            
            # Use SciPy for precise evaluation (unlike the soft approximation used in loss)
            rho, _ = stats.spearmanr(preds, targets)
            r_val, _ = stats.pearsonr(preds, targets)
            
            # Keep original category name as Spearman for backward compatibility
            results[cat] = rho
            results[f"{cat} (Pearson)"] = r_val
        else:
            results[cat] = 0.0
            results[f"{cat} (Pearson)"] = 0.0
            
    return results
