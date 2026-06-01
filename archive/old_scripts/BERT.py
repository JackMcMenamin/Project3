"""
BERT RSR Fine-tuning: Word-Level Similarity Alignment

Fine-tunes pre-trained BERT embeddings using combined RSR datasets
(THINGS + MEN + SimVerb).

Approach (following Mark's implementation):
1. Load pre-trained bert-base-uncased
2. Add projection layer (768 → 128 dim) like Mark did
3. Use CONTEXTUAL embeddings (full BERT forward pass, not just word_embeddings)
4. Freeze embeddings + first 11 layers, only train last layer + projection
5. Apply RSR loss (soft Spearman correlation) to align with human similarity
6. Evaluate on SimLex-999 benchmark

Key differences from previous version:
- Contextual embeddings instead of static word_embeddings
- Projection head reduces dimensionality (better for similarity)
- Only last transformer layer is trainable (prevents catastrophic forgetting)
- Uses torchsort if available for better gradients

Usage:
    python BERT.py
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import scipy.io as sio
from scipy.stats import spearmanr

from transformers import BertModel, BertTokenizer

# Try to import torchsort (Mark's choice for differentiable ranking)
try:
    import torchsort
    HAS_TORCHSORT = True
    print("Using torchsort for differentiable ranking")
except ImportError:
    HAS_TORCHSORT = False
    print("torchsort not available, using custom soft_rank")

# ==============================================================================
# Configuration
# ==============================================================================

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths
BASE_DATA_DIR = Path("data")

# RSR Training datasets
THINGS_DIR = Path("things_similarity")
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"
MEN_PATH = BASE_DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"
SIMVERB_PATH = BASE_DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"

# Evaluation
SIMLEX_PATH = Path("SimLex-999") / "SimLex-999.txt"

# Training hyperparameters
RSR_EPOCHS = 200  # More epochs for combined datasets
RSR_LR = 1e-3  # Higher LR - only training projection + last layer
RSR_SAMPLE_SIZE = 10000  # Number of pairs per RSR step (more with combined data)
SOFT_RANK_STRENGTH = 1.0  # For torchsort (lower = softer)
PROJECTION_DIM = 128  # Mark used 50, we use 128 for more capacity
BATCH_SIZE = 64  # For batched RSR computation

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Random seed: {SEED}")
print(f"Device: {DEVICE}")
print("=" * 70)


# ==============================================================================
# Soft Spearman Correlation (Differentiable) - Following Mark's approach
# ==============================================================================

def soft_rank_custom(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """
    Differentiable soft ranking using pairwise comparisons.
    Fallback when torchsort is not available.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    # Pairwise differences: x_i - x_j
    x_expanded = x.unsqueeze(-1)  # (batch, n, 1)
    x_transposed = x.unsqueeze(-2)  # (batch, 1, n)
    pairwise_diff = x_expanded - x_transposed  # (batch, n, n)
    
    # Soft indicator: sigmoid of scaled difference
    soft_indicator = torch.sigmoid(regularization_strength * pairwise_diff)
    
    # Rank = sum of (how many elements are smaller)
    soft_ranks = soft_indicator.sum(dim=-1)  # (batch, n)
    
    return soft_ranks.squeeze(0) if soft_ranks.shape[0] == 1 else soft_ranks


def soft_spearman(pred: torch.Tensor, target: torch.Tensor, 
                  regularization_strength: float = 1.0) -> torch.Tensor:
    """
    Differentiable Spearman correlation using soft ranking.
    Uses torchsort if available (Mark's approach), otherwise custom implementation.
    
    Args:
        pred: Predicted similarities (n,)
        target: Target similarities (n,)
        regularization_strength: For torchsort, lower = softer ranking
    
    Returns:
        Soft Spearman correlation (scalar)
    """
    # Ensure 2D for torchsort
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
    # Get soft ranks using torchsort or fallback
    if HAS_TORCHSORT:
        pred_rank = torchsort.soft_rank(pred, regularization_strength=regularization_strength)
        target_rank = torchsort.soft_rank(target, regularization_strength=regularization_strength)
    else:
        pred_rank = soft_rank_custom(pred, regularization_strength * 10)  # Scale for custom
        target_rank = soft_rank_custom(target, regularization_strength * 10)
    
    # Center the ranks
    pred_centered = pred_rank - pred_rank.mean()
    target_centered = target_rank - target_rank.mean()
    
    # Pearson correlation of ranks = Spearman correlation
    numerator = (pred_centered * target_centered).sum()
    denominator = pred_centered.norm() * target_centered.norm() + 1e-8
    
    return numerator / denominator


# ==============================================================================
# Load RSR Training Datasets (THINGS, MEN, SimVerb)
# ==============================================================================

def load_men_pairs(path: Path):
    """
    Load MEN dataset (word similarity pairs).
    
    Format: word1-POS word2-POS score
    
    Returns:
        List of (word1, word2, score) tuples
    """
    pairs = []
    
    if not path.exists():
        print(f"  [warn] MEN not found: {path}")
        return pairs
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # Remove POS tag (e.g., "cat-n" -> "cat")
            w1 = parts[0].rsplit("-", 1)[0].lower()
            w2 = parts[1].rsplit("-", 1)[0].lower()
            try:
                score = float(parts[2])
            except:
                continue
            pairs.append((w1, w2, score))
    
    return pairs


def load_simverb_pairs(path: Path):
    """
    Load SimVerb-3500 dataset (verb similarity pairs).
    
    Format: verb1 TAB verb2 TAB POS TAB score TAB ...
    
    Returns:
        List of (word1, word2, score) tuples
    """
    pairs = []
    
    if not path.exists():
        print(f"  [warn] SimVerb not found: {path}")
        return pairs
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            w1 = parts[0].lower()
            w2 = parts[1].lower()
            try:
                score = float(parts[3])
            except:
                continue
            pairs.append((w1, w2, score))
    
    return pairs


def load_things_pairs(words_path: Path, sim_path: Path, max_pairs: int = 50000):
    """
    Load THINGS dataset pairs from similarity matrix.
    
    Returns:
        List of (word1, word2, score) tuples
    """
    pairs = []
    
    if not words_path.exists() or not sim_path.exists():
        print(f"  [warn] THINGS not found")
        return pairs
    
    # Load concept words
    with words_path.open("r", encoding="utf-8") as f:
        things_words = [line.strip().lower().replace("_", " ") for line in f if line.strip()]
    
    # Load similarity matrix
    mat = sio.loadmat(sim_path)
    spose_sim = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if value.shape[0] == value.shape[1] == len(things_words):
                spose_sim = value
                break
    
    if spose_sim is None:
        return pairs
    
    # Generate pairs from upper triangle
    n = len(things_words)
    tri_i, tri_j = np.triu_indices(n, k=1)
    
    total_pairs = len(tri_i)
    if total_pairs > max_pairs:
        sample_idx = np.random.choice(total_pairs, size=max_pairs, replace=False)
        tri_i = tri_i[sample_idx]
        tri_j = tri_j[sample_idx]
    
    for i, j in zip(tri_i, tri_j):
        w1 = things_words[i]
        w2 = things_words[j]
        score = spose_sim[i, j]
        pairs.append((w1, w2, float(score)))
    
    return pairs


def load_all_rsr_datasets():
    """
    Load and combine all RSR training datasets.
    
    Returns:
        all_pairs: List of (word1, word2, normalized_score) tuples
        dataset_info: Dict with info about each dataset
    """
    print("\n" + "=" * 70)
    print("Loading RSR Training Datasets")
    print("=" * 70)
    
    dataset_info = {}
    
    # Load THINGS
    things_pairs = load_things_pairs(THINGS_WORDS_PATH, THINGS_SIM_PATH, max_pairs=50000)
    if things_pairs:
        # Normalize THINGS scores to [0, 1]
        scores = np.array([p[2] for p in things_pairs])
        min_s, max_s = scores.min(), scores.max()
        things_pairs = [(w1, w2, (s - min_s) / (max_s - min_s + 1e-8)) 
                        for w1, w2, s in things_pairs]
    dataset_info['THINGS'] = len(things_pairs)
    print(f"  THINGS:  {len(things_pairs):,} pairs")
    
    # Load MEN
    men_pairs = load_men_pairs(MEN_PATH)
    if men_pairs:
        # Normalize MEN scores to [0, 1] (original is 0-50)
        scores = np.array([p[2] for p in men_pairs])
        min_s, max_s = scores.min(), scores.max()
        men_pairs = [(w1, w2, (s - min_s) / (max_s - min_s + 1e-8)) 
                     for w1, w2, s in men_pairs]
    dataset_info['MEN'] = len(men_pairs)
    print(f"  MEN:     {len(men_pairs):,} pairs")
    
    # Load SimVerb
    simverb_pairs = load_simverb_pairs(SIMVERB_PATH)
    if simverb_pairs:
        # Normalize SimVerb scores to [0, 1] (original is 0-10)
        scores = np.array([p[2] for p in simverb_pairs])
        min_s, max_s = scores.min(), scores.max()
        simverb_pairs = [(w1, w2, (s - min_s) / (max_s - min_s + 1e-8)) 
                         for w1, w2, s in simverb_pairs]
    dataset_info['SimVerb'] = len(simverb_pairs)
    print(f"  SimVerb: {len(simverb_pairs):,} pairs")
    
    # Combine all pairs
    all_pairs = things_pairs + men_pairs + simverb_pairs
    
    # Collect unique words
    all_words = set()
    for w1, w2, _ in all_pairs:
        all_words.add(w1)
        all_words.add(w2)
    
    dataset_info['total_pairs'] = len(all_pairs)
    dataset_info['unique_words'] = len(all_words)
    
    print(f"\n  TOTAL:   {len(all_pairs):,} pairs, {len(all_words):,} unique words")
    
    return all_pairs, dataset_info, all_words


def load_things_data(words_path: Path, sim_path: Path):
    """
    Load THINGS concepts and behavioral similarity matrix.
    
    Returns:
        things_words: List of concept words
        similarity_matrix: (n_concepts, n_concepts) similarity matrix
    """
    print("\n" + "=" * 70)
    print("Loading THINGS Dataset")
    print("=" * 70)
    
    if not words_path.exists():
        raise FileNotFoundError(f"THINGS words file not found: {words_path}")
    if not sim_path.exists():
        raise FileNotFoundError(f"THINGS similarity file not found: {sim_path}")
    
    # Load concept words
    with words_path.open("r", encoding="utf-8") as f:
        things_words = [line.strip().lower() for line in f if line.strip()]
    print(f"  Loaded {len(things_words)} THINGS concepts")
    
    # Load similarity matrix
    mat = sio.loadmat(sim_path)
    similarity_matrix = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if value.shape[0] == value.shape[1] == len(things_words):
                similarity_matrix = value
                print(f"  Found similarity matrix: {key} with shape {value.shape}")
                break
    
    if similarity_matrix is None:
        raise ValueError("Could not find valid similarity matrix in .mat file")
    
    print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    return things_words, similarity_matrix


# ==============================================================================
# BERT Word Embedding Extractor (Mark's Architecture)
# ==============================================================================

class BERTWordEmbeddings(nn.Module):
    """
    Extracts and fine-tunes word embeddings from BERT.
    
    Following Mark's approach:
    1. Use CONTEXTUAL embeddings (full BERT forward pass)
    2. Add projection layer (768 -> projection_dim)
    3. Freeze embeddings + first N layers, only train last layers + projection
    4. For multi-token words, mean-pool the hidden states
    
    This gives much richer representations than static word_embeddings.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 projection_dim: int = 128,
                 num_frozen_layers: int = 11):
        super().__init__()
        
        print("\n" + "=" * 70)
        print("Loading BERT Model (Mark's Architecture)")
        print("=" * 70)
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        self.hidden_size = self.bert.config.hidden_size
        self.num_layers = self.bert.config.num_hidden_layers
        print(f"  Model: {model_name}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Vocab size: {self.bert.config.vocab_size}")
        
        # Freeze embeddings + first N encoder layers (Mark's approach)
        # This prevents catastrophic forgetting while allowing fine-tuning
        print(f"\n  Freezing strategy (like Mark):")
        print(f"    - Embeddings: FROZEN")
        print(f"    - Encoder layers 0-{num_frozen_layers-1}: FROZEN")
        print(f"    - Encoder layer {num_frozen_layers}: TRAINABLE")
        
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first N encoder layers
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                # Keep last layer(s) trainable
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Freeze pooler (we don't use it)
        if self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = False
        
        # Projection layer (Mark used 50, we use 128 for more capacity)
        self.projection_dim = projection_dim
        self.projection = nn.Linear(self.hidden_size, projection_dim)
        print(f"    - Projection layer: {self.hidden_size} -> {projection_dim}: TRAINABLE")
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """
        Get CONTEXTUAL embedding for a single word.
        
        Process:
        1. Tokenize word (may split into subwords)
        2. Add [CLS] and [SEP] tokens
        3. Run through full BERT
        4. Mean-pool the word token hidden states (excluding [CLS] and [SEP])
        5. Project to lower dimension
        """
        device = next(self.parameters()).device
        
        # Tokenize with special tokens
        tokens = self.tokenizer.tokenize(word)
        if not tokens:
            return None
        
        # Build input: [CLS] + word_tokens + [SEP]
        input_tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        token_ids = torch.tensor([token_ids], device=device)  # (1, seq_len)
        attention_mask = torch.ones_like(token_ids)
        
        # Forward through BERT
        outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, 768)
        
        # Mean-pool the word tokens (exclude [CLS] at 0 and [SEP] at -1)
        word_hidden = hidden_states[0, 1:-1, :]  # (num_word_tokens, 768)
        pooled = word_hidden.mean(dim=0)  # (768,)
        
        # Project to lower dimension
        projected = self.projection(pooled)  # (projection_dim,)
        
        return projected
    
    def get_batch_embeddings(self, words: list) -> torch.Tensor:
        """
        Get embeddings for a batch of words.
        More efficient than calling get_word_embedding repeatedly.
        """
        device = next(self.parameters()).device
        
        # Tokenize all words
        all_tokens = []
        word_lengths = []  # Track how many tokens per word (excluding special tokens)
        
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            if not tokens:
                tokens = ['[UNK]']
            all_tokens.append(tokens)
            word_lengths.append(len(tokens))
        
        # Build inputs with padding
        max_len = max(word_lengths) + 2  # +2 for [CLS] and [SEP]
        
        batch_ids = []
        batch_masks = []
        
        for tokens in all_tokens:
            input_tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            
            # Pad to max_len
            padding_len = max_len - len(token_ids)
            attention_mask = [1] * len(token_ids) + [0] * padding_len
            token_ids = token_ids + [0] * padding_len
            
            batch_ids.append(token_ids)
            batch_masks.append(attention_mask)
        
        batch_ids = torch.tensor(batch_ids, device=device)
        batch_masks = torch.tensor(batch_masks, device=device)
        
        # Forward through BERT
        outputs = self.bert(input_ids=batch_ids, attention_mask=batch_masks)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Mean-pool word tokens for each word
        embeddings = []
        for i, length in enumerate(word_lengths):
            # Word tokens are at positions 1 to 1+length (excluding [CLS] and [SEP])
            word_hidden = hidden_states[i, 1:1+length, :]  # (length, 768)
            pooled = word_hidden.mean(dim=0)  # (768,)
            projected = self.projection(pooled)  # (projection_dim,)
            embeddings.append(projected)
        
        return torch.stack(embeddings)  # (batch, projection_dim)


# ==============================================================================
# Align THINGS Concepts with BERT Vocabulary
# ==============================================================================

def align_things_with_bert(things_words: list, tokenizer: BertTokenizer):
    """
    Find which THINGS concepts can be represented in BERT's vocabulary.
    
    Returns:
        aligned_words: List of (things_idx, word, token_ids) tuples
        word_to_things_idx: Dict mapping word -> things matrix index
    """
    print("\n" + "=" * 70)
    print("Aligning THINGS Concepts with BERT Vocabulary")
    print("=" * 70)
    
    aligned = []
    word_to_things_idx = {}
    
    # Track multi-token words
    single_token = 0
    multi_token = 0
    not_found = 0
    
    for idx, word in enumerate(things_words):
        # Try original and variants
        variants = [
            word,
            word.replace("_", " "),
            word.replace("_", ""),
            word.replace(" ", "_"),
        ]
        
        found = False
        for variant in variants:
            tokens = tokenizer.tokenize(variant)
            if tokens:
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                aligned.append((idx, variant, token_ids))
                word_to_things_idx[variant] = idx
                
                if len(tokens) == 1:
                    single_token += 1
                else:
                    multi_token += 1
                found = True
                break
        
        if not found:
            not_found += 1
    
    print(f"  Aligned concepts: {len(aligned)} / {len(things_words)}")
    print(f"    Single-token: {single_token}")
    print(f"    Multi-token:  {multi_token}")
    print(f"    Not found:    {not_found}")
    
    return aligned, word_to_things_idx


# ==============================================================================
# RSR Training Loop (Combined Datasets)
# ==============================================================================

def train_rsr(model: BERTWordEmbeddings, 
              all_pairs: list,
              rsr_words: set,
              n_epochs: int = 200,
              sample_size: int = 5000,
              lr: float = 1e-4):
    """
    Train BERT embeddings with RSR loss using combined datasets.
    
    Args:
        model: BERTWordEmbeddings model
        all_pairs: List of (word1, word2, score) tuples from all datasets
        rsr_words: Set of all words used in RSR training
        n_epochs: Number of training epochs
        sample_size: Number of pairs to sample per epoch
        lr: Learning rate
    
    Returns:
        Dictionary of training history
    """
    print("\n" + "=" * 70)
    print("RSR Training (Combined Datasets)")
    print("=" * 70)
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Sample size per epoch: {sample_size}")
    print(f"  Soft rank strength: {SOFT_RANK_STRENGTH}")
    print(f"  Total training pairs: {len(all_pairs):,}")
    
    model = model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Filter pairs to those where both words can be embedded
    valid_pairs = []
    word_cache = {}  # Cache which words can be embedded
    
    print("\n  Validating pairs against BERT vocabulary...")
    for w1, w2, score in tqdm(all_pairs, desc="Filtering pairs"):
        # Check/cache word validity
        if w1 not in word_cache:
            emb = model.get_word_embedding(w1)
            word_cache[w1] = emb is not None
        if w2 not in word_cache:
            emb = model.get_word_embedding(w2)
            word_cache[w2] = emb is not None
        
        if word_cache[w1] and word_cache[w2]:
            valid_pairs.append((w1, w2, score))
    
    print(f"  Valid pairs: {len(valid_pairs):,} / {len(all_pairs):,}")
    
    if len(valid_pairs) < 100:
        print("  [ERROR] Too few valid pairs for training!")
        return {}, 0, 0
    
    # Convert to numpy for efficient sampling
    pair_words = [(w1, w2) for w1, w2, _ in valid_pairs]
    pair_scores = np.array([s for _, _, s in valid_pairs])
    
    history = {
        'epoch': [],
        'loss': [],
        'spearman_train': [],
        'grad_norm': []
    }
    
    # Initial evaluation on training pairs (sample) - with batching
    model.eval()
    with torch.no_grad():
        eval_idx = np.random.choice(len(valid_pairs), size=min(2000, len(valid_pairs)), replace=False)
        target_sims = pair_scores[eval_idx]
        
        # Batch compute embeddings for evaluation
        eval_words = set()
        for idx in eval_idx:
            w1, w2 = pair_words[idx]
            eval_words.add(w1)
            eval_words.add(w2)
        eval_words_list = list(eval_words)
        
        eval_word_to_emb = {}
        for i in range(0, len(eval_words_list), 64):
            batch_words = eval_words_list[i:i+64]
            batch_embs = model.get_batch_embeddings(batch_words)
            for word, emb in zip(batch_words, batch_embs):
                eval_word_to_emb[word] = emb
        
        model_sims = []
        for idx in eval_idx:
            w1, w2 = pair_words[idx]
            emb1 = eval_word_to_emb[w1]
            emb2 = eval_word_to_emb[w2]
            emb1_norm = emb1 / (emb1.norm() + 1e-8)
            emb2_norm = emb2 / (emb2.norm() + 1e-8)
            sim = (emb1_norm * emb2_norm).sum().item()
            model_sims.append(sim)
        
        initial_rho, _ = spearmanr(model_sims, target_sims)
    
    print(f"\n  Initial training Spearman ρ: {initial_rho:.4f}")
    print("\nTraining...")
    
    # Get all unique words for efficient batching
    all_unique_words = list(set(w for w1, w2 in pair_words for w in (w1, w2)))
    print(f"  Unique words in training set: {len(all_unique_words):,}")
    
    pbar = tqdm(range(n_epochs), desc="RSR Training")
    
    for epoch in pbar:
        model.train()
        
        # Sample pairs for this epoch
        sample_idx = np.random.choice(len(valid_pairs), size=min(sample_size, len(valid_pairs)), replace=False)
        
        # Get unique words in this sample (OPTIMIZATION: batch compute these)
        sample_words_set = set()
        for idx in sample_idx:
            w1, w2 = pair_words[idx]
            sample_words_set.add(w1)
            sample_words_set.add(w2)
        sample_words_list = list(sample_words_set)
        
        # Batch compute embeddings for all unique words at once (MUCH faster!)
        word_to_emb = {}
        batch_size = 64  # Process words in batches for GPU memory
        for i in range(0, len(sample_words_list), batch_size):
            batch_words = sample_words_list[i:i+batch_size]
            batch_embs = model.get_batch_embeddings(batch_words)
            for word, emb in zip(batch_words, batch_embs):
                word_to_emb[word] = emb
        
        # Now build pair embeddings using dictionary lookup (instant!)
        embeddings_1 = []
        embeddings_2 = []
        target_scores = []
        
        for idx in sample_idx:
            w1, w2 = pair_words[idx]
            if w1 in word_to_emb and w2 in word_to_emb:
                embeddings_1.append(word_to_emb[w1])
                embeddings_2.append(word_to_emb[w2])
                target_scores.append(pair_scores[idx])
        
        if len(embeddings_1) < 10:
            continue
        
        embeddings_1 = torch.stack(embeddings_1)
        embeddings_2 = torch.stack(embeddings_2)
        
        # Compute model similarities (cosine)
        emb1_norm = embeddings_1 / (embeddings_1.norm(dim=1, keepdim=True) + 1e-8)
        emb2_norm = embeddings_2 / (embeddings_2.norm(dim=1, keepdim=True) + 1e-8)
        model_sim = (emb1_norm * emb2_norm).sum(dim=1)
        
        # Target similarities
        target_sim = torch.tensor(target_scores, dtype=torch.float32, device=DEVICE)
        
        # RSR Loss: 1 - soft_spearman
        spearman_corr = soft_spearman(model_sim, target_sim, SOFT_RANK_STRENGTH)
        loss = 1.0 - spearman_corr
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        optimizer.step()
        
        # Evaluate every 20 epochs (with batching for speed)
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                eval_idx = np.random.choice(len(valid_pairs), size=min(2000, len(valid_pairs)), replace=False)
                target_sims = pair_scores[eval_idx]
                
                # Batch compute eval embeddings
                eval_words = set()
                for idx in eval_idx:
                    w1, w2 = pair_words[idx]
                    eval_words.add(w1)
                    eval_words.add(w2)
                eval_words_list = list(eval_words)
                
                eval_emb_dict = {}
                for i in range(0, len(eval_words_list), 64):
                    batch = eval_words_list[i:i+64]
                    batch_embs = model.get_batch_embeddings(batch)
                    for w, e in zip(batch, batch_embs):
                        eval_emb_dict[w] = e
                
                model_sims = []
                for idx in eval_idx:
                    w1, w2 = pair_words[idx]
                    emb1, emb2 = eval_emb_dict[w1], eval_emb_dict[w2]
                    emb1_norm = emb1 / (emb1.norm() + 1e-8)
                    emb2_norm = emb2 / (emb2.norm() + 1e-8)
                    sim = (emb1_norm * emb2_norm).sum().item()
                    model_sims.append(sim)
                
                train_rho, _ = spearmanr(model_sims, target_sims)
            
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['spearman_train'].append(train_rho)
            history['grad_norm'].append(grad_norm)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ρ_train': f'{train_rho:.4f}',
                'grad': f'{grad_norm:.2e}'
            })
    
    # Final evaluation (with batching)
    model.eval()
    with torch.no_grad():
        eval_idx = np.random.choice(len(valid_pairs), size=min(5000, len(valid_pairs)), replace=False)
        target_sims = pair_scores[eval_idx]
        
        # Batch compute final eval embeddings
        eval_words = set()
        for idx in eval_idx:
            w1, w2 = pair_words[idx]
            eval_words.add(w1)
            eval_words.add(w2)
        eval_words_list = list(eval_words)
        
        eval_emb_dict = {}
        for i in range(0, len(eval_words_list), 64):
            batch = eval_words_list[i:i+64]
            batch_embs = model.get_batch_embeddings(batch)
            for w, e in zip(batch, batch_embs):
                eval_emb_dict[w] = e
        
        model_sims = []
        for idx in eval_idx:
            w1, w2 = pair_words[idx]
            emb1, emb2 = eval_emb_dict[w1], eval_emb_dict[w2]
            emb1_norm = emb1 / (emb1.norm() + 1e-8)
            emb2_norm = emb2 / (emb2.norm() + 1e-8)
            sim = (emb1_norm * emb2_norm).sum().item()
            model_sims.append(sim)
        
        final_rho, _ = spearmanr(model_sims, target_sims)
    
    print(f"\n  Final training Spearman ρ: {final_rho:.4f}")
    print(f"  Improvement: {final_rho - initial_rho:+.4f}")
    
    return history, initial_rho, final_rho


# ==============================================================================
# SimLex-999 Evaluation
# ==============================================================================

def load_simlex(path: Path):
    """Load SimLex-999 benchmark."""
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                w1 = parts[0].lower()
                w2 = parts[1].lower()
                score = float(parts[3])  # SimLex999 score
                pos = parts[2] if len(parts) > 2 else "N"
                pairs.append((w1, w2, score, pos))
    return pairs


def evaluate_simlex(model: BERTWordEmbeddings, 
                    simlex_pairs: list,
                    rsr_words: set):
    """
    Evaluate BERT embeddings on SimLex-999.
    
    Categorizes pairs by how many words are in RSR training set.
    """
    print("\n" + "=" * 70)
    print("SimLex-999 Evaluation")
    print("=" * 70)
    
    model.eval()
    
    # Categorize pairs by RSR training set membership
    categories = {
        'all': [],
        'both_in_rsr': [],
        'one_in_rsr': [],
        'neither_in_rsr': []
    }
    
    with torch.no_grad():
        for w1, w2, score, pos in simlex_pairs:
            # Get embeddings
            emb1 = model.get_word_embedding(w1)
            emb2 = model.get_word_embedding(w2)
            
            if emb1 is None or emb2 is None:
                continue
            
            # Compute similarity
            emb1_norm = emb1 / (emb1.norm() + 1e-8)
            emb2_norm = emb2 / (emb2.norm() + 1e-8)
            model_sim = (emb1_norm * emb2_norm).sum().item()
            
            # Categorize - check if word is in RSR training set
            w1_in = w1 in rsr_words or w1.replace(" ", "_") in rsr_words or w1.replace("_", " ") in rsr_words
            w2_in = w2 in rsr_words or w2.replace(" ", "_") in rsr_words or w2.replace("_", " ") in rsr_words
            
            pair_data = (w1, w2, score, model_sim, pos)
            categories['all'].append(pair_data)
            
            if w1_in and w2_in:
                categories['both_in_rsr'].append(pair_data)
            elif w1_in or w2_in:
                categories['one_in_rsr'].append(pair_data)
            else:
                categories['neither_in_rsr'].append(pair_data)
    
    # Compute correlations
    results = {}
    
    print(f"\n{'Category':<25} {'N pairs':>10} {'Spearman ρ':>12}")
    print("-" * 50)
    
    for cat_name, pairs in categories.items():
        if len(pairs) < 2:
            results[cat_name] = {'n': len(pairs), 'rho': float('nan')}
            print(f"{cat_name:<25} {len(pairs):>10} {'N/A':>12}")
            continue
        
        human_scores = [p[2] for p in pairs]
        model_scores = [p[3] for p in pairs]
        
        rho, pval = spearmanr(human_scores, model_scores)
        results[cat_name] = {'n': len(pairs), 'rho': rho, 'pval': pval}
        
        print(f"{cat_name:<25} {len(pairs):>10} {rho:>12.4f}")
    
    return results


# ==============================================================================
# Nearest Neighbors Comparison
# ==============================================================================

def nearest_neighbors(model: BERTWordEmbeddings,
                      query_words: list,
                      candidate_words: list,
                      top_k: int = 5):
    """
    Find nearest neighbors for query words among candidates.
    """
    print("\n" + "=" * 70)
    print("Nearest Neighbor Analysis")
    print("=" * 70)
    
    model.eval()
    
    # Get candidate embeddings
    candidate_embeddings = {}
    with torch.no_grad():
        for word in candidate_words:
            emb = model.get_word_embedding(word)
            if emb is not None:
                candidate_embeddings[word] = emb / (emb.norm() + 1e-8)
    
    print(f"  Candidates: {len(candidate_embeddings)}")
    
    # Stack candidate embeddings
    candidate_words_list = list(candidate_embeddings.keys())
    candidate_matrix = torch.stack([candidate_embeddings[w] for w in candidate_words_list])
    
    # Find neighbors for each query
    results = {}
    with torch.no_grad():
        for query in query_words:
            emb = model.get_word_embedding(query)
            if emb is None:
                results[query] = []
                continue
            
            emb_norm = emb / (emb.norm() + 1e-8)
            
            # Compute similarities
            sims = torch.mv(candidate_matrix, emb_norm)
            
            # Get top-k (excluding self if present)
            top_indices = torch.argsort(sims, descending=True)
            neighbors = []
            for idx in top_indices:
                word = candidate_words_list[idx]
                if word != query:
                    neighbors.append((word, sims[idx].item()))
                if len(neighbors) >= top_k:
                    break
            
            results[query] = neighbors
            print(f"\n  '{query}' -> {[n[0] for n in neighbors]}")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("BERT RSR Fine-tuning (Mark's Architecture)")
    print("  - Contextual embeddings (full BERT forward pass)")
    print("  - Projection head (768 -> 128)")
    print("  - Combined datasets (THINGS + MEN + SimVerb)")
    print("=" * 70)
    
    # Load all RSR training datasets
    all_pairs, dataset_info, rsr_words = load_all_rsr_datasets()
    
    # Initialize BERT model (Mark's architecture)
    model = BERTWordEmbeddings(
        model_name="bert-base-uncased",
        projection_dim=PROJECTION_DIM,  # 768 -> 128 projection (Mark used 50)
        num_frozen_layers=11  # Freeze first 11 layers, train last layer + projection
    )
    
    # Load SimLex-999 for evaluation
    simlex_pairs = load_simlex(SIMLEX_PATH)
    print(f"\nLoaded SimLex-999: {len(simlex_pairs)} pairs")
    
    # Evaluate BEFORE RSR training (vanilla BERT)
    print("\n" + "=" * 70)
    print("VANILLA BERT Evaluation (Before RSR)")
    print("=" * 70)
    
    vanilla_results = evaluate_simlex(model, simlex_pairs, rsr_words)
    
    # Get vanilla nearest neighbors - use rsr_words as candidates
    test_words = ['cat', 'dog', 'car', 'hammer', 'apple']
    candidate_words = list(rsr_words)
    vanilla_neighbors = nearest_neighbors(model, test_words, candidate_words)
    
    # Train with RSR using combined datasets
    history, initial_rho, final_rho = train_rsr(
        model=model,
        all_pairs=all_pairs,
        rsr_words=rsr_words,
        n_epochs=RSR_EPOCHS,
        sample_size=RSR_SAMPLE_SIZE,
        lr=RSR_LR
    )
    
    # Evaluate AFTER RSR training
    print("\n" + "=" * 70)
    print("RSR BERT Evaluation (After RSR)")
    print("=" * 70)
    
    rsr_results = evaluate_simlex(model, simlex_pairs, rsr_words)
    
    # Get RSR nearest neighbors
    rsr_neighbors = nearest_neighbors(model, test_words, candidate_words)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Vanilla vs RSR BERT")
    print("=" * 70)
    
    print(f"\n{'Category':<25} {'Vanilla ρ':>12} {'RSR ρ':>12} {'Δ':>10}")
    print("-" * 60)
    
    for cat in ['all', 'both_in_rsr', 'one_in_rsr', 'neither_in_rsr']:
        v_rho = vanilla_results[cat]['rho']
        r_rho = rsr_results[cat]['rho']
        delta = r_rho - v_rho if not (np.isnan(v_rho) or np.isnan(r_rho)) else float('nan')
        
        v_str = f"{v_rho:.4f}" if not np.isnan(v_rho) else "N/A"
        r_str = f"{r_rho:.4f}" if not np.isnan(r_rho) else "N/A"
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        
        print(f"{cat:<25} {v_str:>12} {r_str:>12} {d_str:>10}")
    
    # Neighbor comparison
    print("\n" + "=" * 70)
    print("NEAREST NEIGHBOR COMPARISON")
    print("=" * 70)
    
    for word in test_words:
        print(f"\n=== '{word}' ===")
        v_neighbors = [n[0] for n in vanilla_neighbors.get(word, [])]
        r_neighbors = [n[0] for n in rsr_neighbors.get(word, [])]
        print(f"  VANILLA: {v_neighbors}")
        print(f"  RSR:     {r_neighbors}")
    
    # Save model
    save_path = Path("models") / "bert_rsr_combined.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'rsr_words': list(rsr_words),
        'dataset_info': dataset_info,
        'vanilla_results': vanilla_results,
        'rsr_results': rsr_results,
        'history': history,
        'config': {
            'seed': SEED,
            'epochs': RSR_EPOCHS,
            'lr': RSR_LR,
            'sample_size': RSR_SAMPLE_SIZE,
            'soft_rank_strength': SOFT_RANK_STRENGTH,
            'projection_dim': PROJECTION_DIM,
            'num_frozen_layers': 11,
            'uses_contextual_embeddings': True,
            'uses_torchsort': HAS_TORCHSORT,
        }
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
