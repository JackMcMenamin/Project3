"""
GPT-2 RSR Multi-Seed Experiment

Runs GPT-2 RSR fine-tuning across multiple seeds (1-10) to establish
statistical credibility. Results are saved to Excel and printed at the end.

Usage:
    python run_gpt2_seeds.py
"""

import os
import gc
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import scipy.io as sio
from scipy.stats import spearmanr

from transformers import GPT2Model, GPT2Tokenizer

try:
    import torchsort
    HAS_TORCHSORT = True
except ImportError:
    HAS_TORCHSORT = False

# ==============================================================================
# Configuration
# ==============================================================================

SEEDS = list(range(1, 11))  # Seeds 1-10

# Paths
BASE_DATA_DIR = Path("data")
THINGS_DIR = Path("things_similarity")
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"
MEN_PATH = BASE_DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"
SIMVERB_PATH = BASE_DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
SIMLEX_PATH = Path("SimLex-999") / "SimLex-999.txt"

# Training hyperparameters
RSR_EPOCHS = 200
RSR_LR = 1e-3
RSR_SAMPLE_SIZE = 10000
SOFT_RANK_STRENGTH = 1.0
PROJECTION_DIM = 128
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Helper Functions
# ==============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_rank_custom(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n = x.shape[1]
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    soft_comparisons = torch.sigmoid(diff * regularization_strength)
    ranks = soft_comparisons.sum(dim=2) + 0.5
    return ranks.squeeze(0)


def soft_spearman(pred: torch.Tensor, target: torch.Tensor,
                  regularization_strength: float = SOFT_RANK_STRENGTH) -> torch.Tensor:
    if HAS_TORCHSORT:
        pred_rank = torchsort.soft_rank(pred.unsqueeze(0), regularization_strength=regularization_strength).squeeze(0)
        target_rank = torchsort.soft_rank(target.unsqueeze(0), regularization_strength=regularization_strength).squeeze(0)
    else:
        pred_rank = soft_rank_custom(pred, regularization_strength)
        target_rank = soft_rank_custom(target, regularization_strength)
    
    pred_centered = pred_rank - pred_rank.mean()
    target_centered = target_rank - target_rank.mean()
    
    cov = (pred_centered * target_centered).mean()
    pred_std = pred_centered.std()
    target_std = target_centered.std()
    
    correlation = cov / (pred_std * target_std + 1e-8)
    return correlation


# ==============================================================================
# Model
# ==============================================================================

class GPT2WordEmbeddings(nn.Module):
    def __init__(self, model_name="gpt2", projection_dim=128, num_frozen_layers=11):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 has no pad token
        
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size
        self.projection = nn.Linear(self.hidden_size, projection_dim)
        
        # Freeze layers
        for param in self.gpt2.wte.parameters():
            param.requires_grad = False
        for param in self.gpt2.wpe.parameters():
            param.requires_grad = False
        for i in range(num_frozen_layers):
            for param in self.gpt2.h[i].parameters():
                param.requires_grad = False
        for param in self.gpt2.ln_f.parameters():
            param.requires_grad = False
        
        self.to(DEVICE)
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        word_with_space = " " + word
        tokens = self.tokenizer(word_with_space, return_tensors="pt", add_special_tokens=False)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.gpt2(**tokens)
            hidden_states = outputs.last_hidden_state
        
        word_embedding = hidden_states[0].mean(dim=0)
        projected = self.projection(word_embedding)
        return projected
    
    def get_batch_embeddings(self, words: list, batch_size: int = 64) -> dict:
        embeddings = {}
        words_with_space = [" " + w for w in words]
        
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            batch_texts = words_with_space[i:i+batch_size]
            
            tokens = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, add_special_tokens=False)
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.gpt2(**tokens)
                hidden_states = outputs.last_hidden_state
            
            attention_mask = tokens['attention_mask']
            for j, word in enumerate(batch_words):
                mask = attention_mask[j].bool()
                word_hidden = hidden_states[j, mask, :]
                word_embedding = word_hidden.mean(dim=0)
                projected = self.projection(word_embedding)
                embeddings[word] = projected
        
        return embeddings


# ==============================================================================
# Data Loading
# ==============================================================================

def load_men_pairs():
    pairs = []
    if not MEN_PATH.exists():
        return pairs
    with open(MEN_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                w1, w2 = parts[0].split('-')[0], parts[1].split('-')[0]
                score = float(parts[2])
                pairs.append((w1, w2, score))
    return pairs


def load_simverb_pairs():
    pairs = []
    if not SIMVERB_PATH.exists():
        return pairs
    with open(SIMVERB_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                w1, w2 = parts[0], parts[1]
                score = float(parts[3])
                pairs.append((w1, w2, score))
    return pairs


def load_things_pairs():
    pairs = []
    words = []
    with open(THINGS_WORDS_PATH, 'r') as f:
        for line in f:
            word = line.strip().replace(' ', '_')
            words.append(word)
    
    mat_data = sio.loadmat(str(THINGS_SIM_PATH))
    sim_matrix = mat_data['spose_sim']
    
    n = len(words)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((words[i], words[j], float(sim_matrix[i, j])))
    
    return pairs


def load_all_rsr_datasets():
    men_pairs = load_men_pairs()
    simverb_pairs = load_simverb_pairs()
    things_pairs = load_things_pairs()
    
    def normalize(pairs):
        if not pairs:
            return pairs
        scores = [p[2] for p in pairs]
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-8:
            return pairs
        return [(p[0], p[1], (p[2] - min_s) / (max_s - min_s)) for p in pairs]
    
    men_pairs = normalize(men_pairs)
    simverb_pairs = normalize(simverb_pairs)
    things_pairs = normalize(things_pairs)
    
    all_pairs = men_pairs + simverb_pairs + things_pairs
    
    rsr_words = set()
    for w1, w2, _ in all_pairs:
        rsr_words.add(w1)
        rsr_words.add(w2)
    
    dataset_info = {
        'men': len(men_pairs),
        'simverb': len(simverb_pairs),
        'things': len(things_pairs),
        'total': len(all_pairs),
        'unique_words': len(rsr_words)
    }
    
    return all_pairs, dataset_info, rsr_words


def load_simlex(path):
    pairs = []
    with open(path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            w1, w2, score = parts[0], parts[1], float(parts[3])
            pairs.append((w1, w2, score))
    return pairs


# ==============================================================================
# Training and Evaluation
# ==============================================================================

def train_rsr(model, all_pairs, rsr_words, n_epochs, sample_size, lr):
    tokenizer = model.tokenizer
    valid_pairs = []
    for w1, w2, score in all_pairs:
        tokens1 = tokenizer.tokenize(" " + w1)
        tokens2 = tokenizer.tokenize(" " + w2)
        if tokens1 and tokens2:
            valid_pairs.append((w1, w2, score))
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        
        if len(valid_pairs) > sample_size:
            sample = random.sample(valid_pairs, sample_size)
        else:
            sample = valid_pairs
        
        unique_words = list(set([p[0] for p in sample] + [p[1] for p in sample]))
        word_to_emb = model.get_batch_embeddings(unique_words, batch_size=BATCH_SIZE)
        
        model_sims = []
        human_sims = []
        
        for w1, w2, score in sample:
            if w1 in word_to_emb and w2 in word_to_emb:
                emb1 = word_to_emb[w1]
                emb2 = word_to_emb[w2]
                cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                model_sims.append(cos_sim)
                human_sims.append(score)
        
        if len(model_sims) < 10:
            continue
        
        model_sims_tensor = torch.cat(model_sims)
        human_sims_tensor = torch.tensor(human_sims, device=DEVICE)
        
        rho = soft_spearman(model_sims_tensor, human_sims_tensor)
        loss = 1 - rho
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, rho={rho.item():.4f}")
    
    return {}


def evaluate_simlex(model, simlex_pairs, rsr_words):
    model.eval()
    
    categories = defaultdict(list)
    for w1, w2, score in simlex_pairs:
        in1 = w1 in rsr_words
        in2 = w2 in rsr_words
        
        if in1 and in2:
            cat = 'both_in_rsr'
        elif in1 or in2:
            cat = 'one_in_rsr'
        else:
            cat = 'neither_in_rsr'
        
        categories['all'].append((w1, w2, score))
        categories[cat].append((w1, w2, score))
    
    results = {}
    
    for cat_name, pairs in categories.items():
        all_words = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        
        with torch.no_grad():
            word_to_emb = model.get_batch_embeddings(all_words, batch_size=BATCH_SIZE)
        
        model_scores = []
        human_scores = []
        
        for w1, w2, score in pairs:
            if w1 not in word_to_emb or w2 not in word_to_emb:
                continue
            
            emb1 = word_to_emb[w1]
            emb2 = word_to_emb[w2]
            cos_sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
            
            model_scores.append(cos_sim)
            human_scores.append(score)
        
        if len(model_scores) < 2:
            results[cat_name] = {'n': 0, 'rho': float('nan')}
            continue
        
        rho, _ = spearmanr(human_scores, model_scores)
        results[cat_name] = {'n': len(model_scores), 'rho': rho}
    
    return results


# ==============================================================================
# Main Multi-Seed Experiment
# ==============================================================================

def run_single_seed(seed, all_pairs, rsr_words, simlex_pairs):
    """Run a single seed experiment and return results."""
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    
    set_seed(seed)
    
    # Initialize model
    model = GPT2WordEmbeddings(
        model_name="gpt2",
        projection_dim=PROJECTION_DIM,
        num_frozen_layers=11
    )
    
    # Evaluate vanilla
    print("  Evaluating vanilla GPT-2...")
    vanilla_results = evaluate_simlex(model, simlex_pairs, rsr_words)
    
    # Train RSR
    print("  Training RSR...")
    train_rsr(model, all_pairs, rsr_words, RSR_EPOCHS, RSR_SAMPLE_SIZE, RSR_LR)
    
    # Evaluate RSR
    print("  Evaluating RSR GPT-2...")
    rsr_results = evaluate_simlex(model, simlex_pairs, rsr_words)
    
    # Collect results
    result = {
        'seed': seed,
        'vanilla_all': vanilla_results['all']['rho'],
        'vanilla_both': vanilla_results['both_in_rsr']['rho'],
        'vanilla_one': vanilla_results['one_in_rsr']['rho'],
        'vanilla_neither': vanilla_results['neither_in_rsr']['rho'],
        'rsr_all': rsr_results['all']['rho'],
        'rsr_both': rsr_results['both_in_rsr']['rho'],
        'rsr_one': rsr_results['one_in_rsr']['rho'],
        'rsr_neither': rsr_results['neither_in_rsr']['rho'],
    }
    
    # Calculate deltas
    result['delta_all'] = result['rsr_all'] - result['vanilla_all']
    result['delta_both'] = result['rsr_both'] - result['vanilla_both']
    result['delta_one'] = result['rsr_one'] - result['vanilla_one']
    result['delta_neither'] = result['rsr_neither'] - result['vanilla_neither']
    
    # Print seed result
    print(f"\n  Seed {seed} Results:")
    print(f"    All:     Vanilla={result['vanilla_all']:.4f}, RSR={result['rsr_all']:.4f}, Δ={result['delta_all']:+.4f}")
    print(f"    Neither: Vanilla={result['vanilla_neither']:.4f}, RSR={result['rsr_neither']:.4f}, Δ={result['delta_neither']:+.4f}")
    
    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def main():
    print("=" * 70)
    print("GPT-2 RSR Multi-Seed Experiment")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print(f"Torchsort: {HAS_TORCHSORT}")
    print("=" * 70)
    
    # Load data once
    print("\nLoading datasets...")
    all_pairs, dataset_info, rsr_words = load_all_rsr_datasets()
    simlex_pairs = load_simlex(SIMLEX_PATH)
    
    print(f"  Total RSR pairs: {dataset_info['total']}")
    print(f"  RSR vocabulary: {dataset_info['unique_words']} words")
    print(f"  SimLex-999 pairs: {len(simlex_pairs)}")
    
    # Run experiments
    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, all_pairs, rsr_words, simlex_pairs)
        all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = Path("results") / f"gpt2_seeds_{timestamp}.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)
    print(f"\nResults saved to: {excel_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("GPT-2 RSR MULTI-SEED RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("Individual Seed Results:")
    print("-" * 70)
    print(f"{'Seed':<6} {'Van All':>10} {'RSR All':>10} {'Δ All':>10} {'Δ Neither':>12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['seed']:<6} {r['vanilla_all']:>10.4f} {r['rsr_all']:>10.4f} {r['delta_all']:>+10.4f} {r['delta_neither']:>+12.4f}")
    
    print("\n" + "-" * 70)
    print("Aggregated Statistics (Mean ± Std):")
    print("-" * 70)
    
    metrics = [
        ('All pairs - Vanilla', 'vanilla_all'),
        ('All pairs - RSR', 'rsr_all'),
        ('All pairs - Delta', 'delta_all'),
        ('Both in RSR - Delta', 'delta_both'),
        ('One in RSR - Delta', 'delta_one'),
        ('Neither in RSR - Delta', 'delta_neither'),
    ]
    
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for name, col in metrics:
        values = df[col].dropna()
        if len(values) > 0:
            print(f"{name:<30} {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")
    
    # Highlight key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    
    delta_all = df['delta_all'].dropna()
    delta_neither = df['delta_neither'].dropna()
    
    print(f"\nOverall improvement (All pairs):")
    print(f"  Mean Δρ = {delta_all.mean():+.4f} ± {delta_all.std():.4f}")
    
    print(f"\nGeneralization (Neither in RSR):")
    print(f"  Mean Δρ = {delta_neither.mean():+.4f} ± {delta_neither.std():.4f}")
    
    # Statistical significance check
    if delta_all.mean() > 2 * delta_all.std():
        print(f"\n✓ Overall improvement appears robust (mean > 2×std)")
    if delta_neither.mean() > 0:
        print(f"✓ Generalization to unseen words observed in {(delta_neither > 0).sum()}/{len(delta_neither)} seeds")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
