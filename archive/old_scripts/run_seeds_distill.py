"""
Two-Stage RSR Training with Similarity Distillation.

The generalization problem: RSR only updates embeddings for words IN training pairs.
Words NOT in pairs get zero gradient signal.

Solution: Two-stage training with knowledge distillation.

STAGE 1 - Standard RSR Training:
    - Train W2V + RSR on the training pairs (MEN, SimVerb, THINGS)
    - This gives us embeddings where TRAINED words have human-aligned similarities
    
STAGE 2 - Similarity Distillation:
    - For each UNSEEN word W (not in training vocab):
        1. Find its K nearest neighbors among TRAINED words: {A1, A2, ..., Ak}
        2. Compute interpolation weights based on cosine similarity
        3. For target word X, estimate: pseudo_sim(W, X) = weighted_avg(sim(Ai, X))
        4. Train W's embedding to match these pseudo-similarities
    
    This "distills" the RSR knowledge from trained words to unseen words.

Theory: If W is semantically similar to trained word A (by distributional semantics),
then W's relationships to other words should mirror A's relationships.

Usage:
    python run_seeds_distill.py
"""

import os
import re
import json
import random
import argparse
from copy import deepcopy
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy.io as sio
from scipy.stats import spearmanr

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DATA_DIR = Path("data")
WIKI_DIR = BASE_DATA_DIR / "enwiki_namespace_0"

# RSR training datasets
MEN_PATH = BASE_DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"
SIMVERB_PATH = BASE_DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
THINGS_DIR = Path("things_similarity")
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"

# SimLex-999 for evaluation
SIMLEX_PATH = Path("SimLex-999") / "SimLex-999.txt"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
MIN_COUNT = 30
MAX_VOCAB = None
EMBEDDING_DIM = 300
WINDOW_SIZE = 5
NEG_SAMPLES = 5
SUBSAMPLE_T = 1e-5

# Stage 1: Standard training
EPOCHS_STAGE1 = 2
BATCH_SIZE = 8192
LR = 2e-3
BATCHES_PER_EPOCH = 10000
RSR_WEIGHT = 0.5
RSR_PAIRS_PER_STEP = 5000
SOFT_RANK_STRENGTH = 2.0

# Stage 2: Distillation (VECTORIZED for speed)
EPOCHS_STAGE2 = 1
DISTILL_BATCH_SIZE = 256  # Number of unseen words per batch
DISTILL_K_NEIGHBORS = 10  # Number of trained neighbors to interpolate from
DISTILL_N_TARGETS = 100  # Number of shared target words per batch
DISTILL_BATCHES = 2000  # Batches for distillation
DISTILL_LR = 1e-3

MAX_JSON_FILES = None
MAX_ARTICLES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Tokenization + streaming (same as before)
# ==============================================================================

_token_re = re.compile(r"[^a-zA-Z\s]+")

def simple_tokenize(text: str):
    text = text.lower()
    text = _token_re.sub(" ", text)
    return text.split()

def iter_wiki_sentences_jsonl(wiki_dir: Path, max_files=None, max_articles=None):
    jsonl_files = sorted(wiki_dir.glob("*.jsonl"))
    if max_files is not None:
        jsonl_files = jsonl_files[:max_files]

    article_count = 0
    for jf in tqdm(jsonl_files, desc="Processing JSONL files", leave=False):
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                if max_articles is not None and article_count >= max_articles:
                    return
                try:
                    art = json.loads(line)
                except:
                    continue
                article_count += 1
                for section in art.get("sections", []):
                    for part in section.get("has_parts", []):
                        if part.get("type") == "paragraph":
                            text = part.get("value", "")
                            if text:
                                for sent in text.split(". "):
                                    toks = simple_tokenize(sent)
                                    if len(toks) >= 2:
                                        yield toks

# ==============================================================================
# Vocab building
# ==============================================================================

def build_vocab_from_stream(stream, min_count=5, max_vocab=None):
    counts = Counter()
    for toks in tqdm(stream, desc="Counting vocab (stream pass 1)"):
        counts.update(toks)

    items = [(w, c) for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    if max_vocab is not None:
        items = items[:max_vocab]

    vocab = ["<UNK>"] + [w for w, _ in items]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    idx_counts = np.zeros(len(vocab), dtype=np.int64)
    idx_counts[0] = 1
    for w, c in items:
        idx_counts[word2idx[w]] = c

    return word2idx, idx2word, idx_counts

# ==============================================================================
# Sampling distributions
# ==============================================================================

def make_unigram_dist(idx_counts: np.ndarray, power: float = 0.75) -> torch.Tensor:
    freqs = idx_counts.astype(np.float64)
    freqs[0] = 0.0
    p = np.power(freqs, power)
    p = p / (p.sum() + 1e-12)
    return torch.tensor(p, dtype=torch.float32)

def make_subsampling_keep_probs(idx_counts: np.ndarray, t: float = 1e-5) -> np.ndarray:
    freqs = idx_counts / idx_counts.sum()
    keep = np.ones_like(freqs, dtype=np.float64)
    mask = freqs > 0
    keep[mask] = np.minimum(1.0, (np.sqrt(t / freqs[mask]) + (t / freqs[mask])))
    keep[0] = 0.0
    return keep

# ==============================================================================
# Skip-gram pairs
# ==============================================================================

def iter_skipgram_pairs(sentence_stream, window_size, word2idx, keep_prob):
    for toks in sentence_stream:
        idxs = []
        for w in toks:
            i = word2idx.get(w, 0)
            if i == 0:
                continue
            if keep_prob is not None:
                if random.random() > keep_prob[i]:
                    continue
            idxs.append(i)
        if len(idxs) < 2:
            continue
        for center_pos, target in enumerate(idxs):
            left = max(0, center_pos - window_size)
            right = min(len(idxs), center_pos + window_size + 1)
            for ctx_pos in range(left, right):
                if ctx_pos == center_pos:
                    continue
                yield target, idxs[ctx_pos]

def batch_pairs(pair_iter, batch_size):
    targets = []
    contexts = []
    for t, c in pair_iter:
        targets.append(t)
        contexts.append(c)
        if len(targets) >= batch_size:
            yield torch.tensor(targets, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)
            targets, contexts = [], []

# ==============================================================================
# Model
# ==============================================================================

class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.uniform_(self.out_embed.weight, -bound, bound)

    def forward(self, target_idx, pos_ctx_idx, neg_ctx_idx):
        v = self.in_embed(target_idx)
        u_pos = self.out_embed(pos_ctx_idx)
        u_neg = self.out_embed(neg_ctx_idx)
        pos_logits = (v * u_pos).sum(dim=1)
        neg_logits = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        return pos_logits, neg_logits

def w2v_neg_sampling_loss(pos_logits, neg_logits):
    pos_loss = F.logsigmoid(pos_logits).mean()
    neg_loss = F.logsigmoid(-neg_logits).mean()
    return -(pos_loss + neg_loss)

# ==============================================================================
# RSR (soft rank + soft Spearman)
# ==============================================================================

def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    x = x.flatten()
    diffs = x.unsqueeze(1) - x.unsqueeze(0)
    soft_comparisons = torch.sigmoid(regularization_strength * diffs)
    ranks = soft_comparisons.sum(dim=1)
    return ranks

def soft_spearman(pred: torch.Tensor, target: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    pr = soft_rank(pred, regularization_strength)
    tr = soft_rank(target, regularization_strength)
    pr = pr - pr.mean()
    tr = tr - tr.mean()
    pr = pr / (pr.norm() + 1e-8)
    tr = tr / (tr.norm() + 1e-8)
    return (pr * tr).sum()

# ==============================================================================
# Multi-Dataset Loader
# ==============================================================================

def load_men_pairs(path: Path, word2idx: dict):
    pairs = []
    words = set()
    
    if not path.exists():
        print(f"[warn] MEN not found: {path}")
        return words, pairs
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            w1 = parts[0].rsplit("-", 1)[0].lower()
            w2 = parts[1].rsplit("-", 1)[0].lower()
            try:
                score = float(parts[2])
            except:
                continue
            idx1 = word2idx.get(w1, 0)
            idx2 = word2idx.get(w2, 0)
            if idx1 != 0 and idx2 != 0:
                words.add(w1)
                words.add(w2)
                pairs.append((idx1, idx2, score))
    
    return words, pairs

def load_simverb_pairs(path: Path, word2idx: dict):
    pairs = []
    words = set()
    
    if not path.exists():
        print(f"[warn] SimVerb not found: {path}")
        return words, pairs
    
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
            idx1 = word2idx.get(w1, 0)
            idx2 = word2idx.get(w2, 0)
            if idx1 != 0 and idx2 != 0:
                words.add(w1)
                words.add(w2)
                pairs.append((idx1, idx2, score))
    
    return words, pairs

def load_things_pairs(words_path: Path, sim_path: Path, word2idx: dict, max_pairs: int = 50000):
    pairs = []
    words = set()
    
    if not words_path.exists() or not sim_path.exists():
        print(f"[warn] THINGS not found")
        return words, pairs
    
    with words_path.open("r", encoding="utf-8") as f:
        things_words = [ln.strip().lower().replace("_", " ") for ln in f if ln.strip()]
    
    mat = sio.loadmat(sim_path)
    spose_sim = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == v.shape[1]:
            spose_sim = v
            break
    
    if spose_sim is None:
        return words, pairs
    
    valid_indices = []
    valid_things_indices = []
    for ti, tw in enumerate(things_words):
        for variant in [tw, tw.replace(" ", "")]:
            vi = word2idx.get(variant, 0)
            if vi != 0:
                words.add(variant)
                valid_indices.append(vi)
                valid_things_indices.append(ti)
                break
    
    if len(valid_indices) < 2:
        return words, pairs
    
    valid_indices = np.array(valid_indices)
    valid_things_indices = np.array(valid_things_indices)
    
    n = len(valid_indices)
    tri_i, tri_j = np.triu_indices(n, k=1)
    
    total_pairs = len(tri_i)
    if total_pairs > max_pairs:
        sample_idx = np.random.choice(total_pairs, size=max_pairs, replace=False)
        tri_i = tri_i[sample_idx]
        tri_j = tri_j[sample_idx]
    
    for i, j in zip(tri_i, tri_j):
        ti = valid_things_indices[i]
        tj = valid_things_indices[j]
        score = spose_sim[ti, tj]
        pairs.append((valid_indices[i], valid_indices[j], float(score)))
    
    return words, pairs

def load_all_datasets(word2idx: dict):
    print("\n--- Loading RSR Training Datasets ---")
    
    men_words, men_pairs = load_men_pairs(MEN_PATH, word2idx)
    print(f"  MEN:     {len(men_pairs):,} pairs, {len(men_words):,} unique words")
    
    simverb_words, simverb_pairs = load_simverb_pairs(SIMVERB_PATH, word2idx)
    print(f"  SimVerb: {len(simverb_pairs):,} pairs, {len(simverb_words):,} unique words")
    
    things_words, things_pairs = load_things_pairs(THINGS_WORDS_PATH, THINGS_SIM_PATH, word2idx, max_pairs=50000)
    print(f"  THINGS:  {len(things_pairs):,} pairs, {len(things_words):,} unique words")
    
    all_words = men_words | simverb_words | things_words
    print(f"  TOTAL:   {len(all_words):,} unique words in training vocab")
    
    all_pairs = []
    for pairs, name in [(men_pairs, "MEN"), (simverb_pairs, "SimVerb"), (things_pairs, "THINGS")]:
        if len(pairs) == 0:
            continue
        arr = np.array(pairs, dtype=np.float32)
        scores = arr[:, 2]
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            arr[:, 2] = (scores - min_s) / (max_s - min_s)
        all_pairs.append(arr)
    
    if len(all_pairs) == 0:
        raise ValueError("No similarity pairs loaded!")
    
    combined = np.vstack(all_pairs)
    print(f"  Combined: {len(combined):,} total pairs for RSR training")
    
    # Get indices of trained words
    trained_indices = set()
    for pairs in [men_pairs, simverb_pairs, things_pairs]:
        for idx1, idx2, _ in pairs:
            trained_indices.add(int(idx1))
            trained_indices.add(int(idx2))
    
    return all_words, combined, trained_indices

def sample_rsr_pairs(num_pairs: int, pairs_array: np.ndarray):
    n_available = len(pairs_array)
    if num_pairs > n_available:
        idx = np.random.randint(0, n_available, size=num_pairs)
    else:
        idx = np.random.choice(n_available, size=num_pairs, replace=False)
    
    sampled = pairs_array[idx]
    
    return (
        torch.tensor(sampled[:, 0].astype(np.int64), dtype=torch.long, device=DEVICE),
        torch.tensor(sampled[:, 1].astype(np.int64), dtype=torch.long, device=DEVICE),
        torch.tensor(sampled[:, 2], dtype=torch.float32, device=DEVICE),
    )

# ==============================================================================
# STAGE 1: Standard RSR Training
# ==============================================================================

def cosine_sim_from_in_embeddings_grad(model, idx_a, idx_b):
    va = model.in_embed(idx_a)
    vb = model.in_embed(idx_b)
    va = va / (va.norm(dim=1, keepdim=True) + 1e-8)
    vb = vb / (vb.norm(dim=1, keepdim=True) + 1e-8)
    return (va * vb).sum(dim=1)

def train_one_epoch_stage1(
    model, optimizer, sentence_stream_fn, batches_per_epoch, batch_size,
    window_size, neg_samples, neg_dist, word2idx, keep_prob,
    rsr_weight=0.5, rsr_pairs_per_step=5000, soft_rank_strength=2.0,
    pairs_array=None
):
    """Stage 1: Standard W2V + RSR training."""
    model.train()

    pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size, word2idx, keep_prob)
    batch_iter = batch_pairs(pair_iter, batch_size)

    total_loss = 0.0
    total_w2v = 0.0
    total_rsr = 0.0

    pbar = tqdm(range(batches_per_epoch), desc="Stage1", leave=False)
    for b in pbar:
        try:
            tgt, ctx = next(batch_iter)
        except StopIteration:
            pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size, word2idx, keep_prob)
            batch_iter = batch_pairs(pair_iter, batch_size)
            tgt, ctx = next(batch_iter)

        tgt = tgt.to(DEVICE)
        ctx = ctx.to(DEVICE)

        neg = torch.multinomial(neg_dist, num_samples=tgt.shape[0] * neg_samples, replacement=True)
        neg = neg.view(tgt.shape[0], neg_samples).to(DEVICE)

        pos_logits, neg_logits = model(tgt, ctx, neg)
        w2v_loss = w2v_neg_sampling_loss(pos_logits, neg_logits)
        total_w2v += float(w2v_loss.item())

        # RSR loss
        if pairs_array is not None and rsr_weight > 0:
            i_idx, j_idx, target_sim = sample_rsr_pairs(rsr_pairs_per_step, pairs_array)
            pred_sim = cosine_sim_from_in_embeddings_grad(model, i_idx, j_idx)
            rho = soft_spearman(pred_sim, target_sim, regularization_strength=soft_rank_strength)
            rsr_loss = 1.0 - rho
            total_rsr += float(rsr_loss.item())
            loss = (1 - rsr_weight) * w2v_loss + rsr_weight * rsr_loss
        else:
            loss = w2v_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return {
        "loss": total_loss / batches_per_epoch,
        "w2v": total_w2v / batches_per_epoch,
        "rsr": total_rsr / batches_per_epoch if pairs_array is not None else 0.0,
    }

# ==============================================================================
# STAGE 2: Similarity Distillation to Unseen Words
# ==============================================================================

def distillation_loss_vectorized(model, unseen_tensor, trained_tensor, target_tensor, 
                                   k_neighbors, temperature=5.0):
    """
    FULLY VECTORIZED distillation loss.
    
    IMPORTANT: Only unseen word embeddings receive gradients!
    Target embeddings are DETACHED to preserve RSR-trained embeddings.
    
    Args:
        unseen_tensor: [batch_size] indices of unseen words
        trained_tensor: [n_trained] indices of all trained words  
        target_tensor: [n_targets] indices of target words (shared across batch)
        k_neighbors: number of nearest trained neighbors to use
        temperature: softmax temperature for neighbor weights
    
    Returns:
        loss: scalar MSE loss
    """
    W = model.in_embed.weight
    batch_size = unseen_tensor.shape[0]
    
    # Get normalized embeddings (ALL detached for neighbor/target computation)
    with torch.no_grad():
        W_detached = W.detach()
        Wn = W_detached / (W_detached.norm(dim=1, keepdim=True) + 1e-8)
        
        # Embeddings for neighbor search
        unseen_emb_detached = Wn[unseen_tensor]  # [batch, dim]
        trained_emb = Wn[trained_tensor]          # [n_trained, dim]
        target_emb_detached = Wn[target_tensor]   # [n_targets, dim] - FROZEN!
        
        # Similarities: unseen -> trained [batch, n_trained]
        sims_to_trained = unseen_emb_detached @ trained_emb.T
        
        # Top-k neighbors for each unseen word [batch, k]
        topk_sims, topk_idx = torch.topk(sims_to_trained, k_neighbors, dim=1)
        neighbor_weights = F.softmax(topk_sims * temperature, dim=1)  # [batch, k]
        
        # Get neighbor global indices [batch, k]
        neighbor_global_idx = trained_tensor[topk_idx]  # [batch, k]
        
        # Neighbor embeddings [batch, k, dim]
        neighbor_emb = Wn[neighbor_global_idx]
        
        # Neighbor -> target similarities [batch, k, n_targets]
        neighbor_to_target = torch.bmm(
            neighbor_emb, 
            target_emb_detached.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)
        )
        
        # Weighted pseudo-similarities [batch, n_targets]
        pseudo_sims = (neighbor_weights.unsqueeze(2) * neighbor_to_target).sum(dim=1)
    
    # ONLY unseen embeddings get gradients
    unseen_emb_grad = W[unseen_tensor]  # [batch, dim] - WITH gradient
    unseen_emb_norm = unseen_emb_grad / (unseen_emb_grad.norm(dim=1, keepdim=True) + 1e-8)
    
    # Target embeddings are DETACHED - no gradient to trained words!
    # This preserves the RSR-aligned embeddings from Stage 1
    actual_sims = unseen_emb_norm @ target_emb_detached.T  # [batch, n_targets]
    
    # MSE loss - only backprops to unseen_emb_grad
    loss = F.mse_loss(actual_sims, pseudo_sims)
    
    return loss

def train_stage2_distillation(model, optimizer, trained_indices_list, vocab_size, 
                              distill_batches, distill_batch_size, k_neighbors, n_targets):
    """
    Stage 2: Train unseen words to match interpolated similarities from trained neighbors.
    FULLY VECTORIZED for speed.
    """
    model.train()
    
    # Get list of unseen word indices (not in training set, not UNK)
    trained_set = set(trained_indices_list)
    unseen_indices = [i for i in range(1, vocab_size) if i not in trained_set]
    
    print(f"  Stage 2: {len(unseen_indices):,} unseen words to distill knowledge to")
    
    if len(unseen_indices) == 0:
        print("  No unseen words - skipping stage 2")
        return {"loss": 0.0}
    
    # Pre-convert to tensors
    trained_tensor = torch.tensor(trained_indices_list, dtype=torch.long, device=DEVICE)
    all_vocab_indices = list(range(1, vocab_size))
    
    total_loss = 0.0
    pbar = tqdm(range(distill_batches), desc="Stage2-Distill", leave=False)
    
    for b in pbar:
        # Sample batch of unseen words
        batch_idx = random.sample(unseen_indices, min(distill_batch_size, len(unseen_indices)))
        unseen_tensor = torch.tensor(batch_idx, dtype=torch.long, device=DEVICE)
        
        # Sample shared target words (mix of trained and random)
        n_trained_tgt = n_targets // 2
        n_random_tgt = n_targets - n_trained_tgt
        target_idx = (random.sample(trained_indices_list, min(n_trained_tgt, len(trained_indices_list))) +
                      random.sample(all_vocab_indices, n_random_tgt))
        target_tensor = torch.tensor(target_idx, dtype=torch.long, device=DEVICE)
        
        loss = distillation_loss_vectorized(
            model, unseen_tensor, trained_tensor, target_tensor, k_neighbors
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.item())
        
        if b % 200 == 0:
            pbar.set_postfix({"loss": f"{total_loss / (b+1):.4f}"})
    
    return {"loss": total_loss / distill_batches}

# ==============================================================================
# Evaluation
# ==============================================================================

def load_simlex(path: Path):
    df = pd.read_csv(path, sep=None, engine="python")
    cols = [c.lower() for c in df.columns]

    if "word1" in cols and "word2" in cols:
        w1_col = df.columns[cols.index("word1")]
        w2_col = df.columns[cols.index("word2")]
    else:
        w1_col, w2_col = df.columns[:2]

    score_col = None
    for candidate in ["simlex999", "simlex", "score", "similarity"]:
        if candidate in cols:
            score_col = df.columns[cols.index(candidate)]
            break
    if score_col is None:
        score_col = df.columns[2]

    df = df[[w1_col, w2_col, score_col]].copy()
    df.columns = ["word1", "word2", "score"]
    df["word1"] = df["word1"].astype(str).str.lower()
    df["word2"] = df["word2"].astype(str).str.lower()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    return df

@torch.no_grad()
def simlex_spearman_by_train_count(model, simlex_df, word2idx, train_words, train_count=None):
    model.eval()
    W = model.in_embed.weight.detach()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    sims = []
    scores = []
    covered = 0

    for _, row in simlex_df.iterrows():
        w1 = row["word1"]
        w2 = row["word2"]

        w1_in_train = w1 in train_words
        w2_in_train = w2 in train_words
        pair_count = int(w1_in_train) + int(w2_in_train)

        if train_count is not None and pair_count != train_count:
            continue

        i = word2idx.get(w1, 0)
        j = word2idx.get(w2, 0)
        if i == 0 or j == 0:
            continue

        s = float((Wn[i] * Wn[j]).sum().item())
        sims.append(s)
        scores.append(float(row["score"]))
        covered += 1

    if covered < 10:
        return {"rho": np.nan, "n": covered}

    rho, _ = spearmanr(sims, scores)
    return {"rho": float(rho), "n": int(covered)}

# ==============================================================================
# Main training pipeline for one seed
# ==============================================================================

def run_one_seed(seed: int, word2idx, idx2word, idx_counts, train_words, pairs_array, 
                 trained_indices, simlex_df):
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    vocab_size = len(word2idx)
    trained_indices_list = list(trained_indices)
    
    neg_dist = make_unigram_dist(idx_counts)
    keep_prob = make_subsampling_keep_probs(idx_counts, t=SUBSAMPLE_T) if SUBSAMPLE_T else None
    
    def sentence_stream_factory():
        return iter_wiki_sentences_jsonl(WIKI_DIR, max_files=MAX_JSON_FILES, max_articles=MAX_ARTICLES)
    
    # Shared init
    init_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    init_state = deepcopy(init_model.state_dict())
    del init_model
    
    # =========================================================================
    # Train Vanilla (pure W2V - no RSR, no distillation)
    # =========================================================================
    print(f"[seed={seed}] Training Vanilla (pure W2V)...")
    vanilla_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    vanilla_model.load_state_dict(deepcopy(init_state))
    vanilla_opt = optim.Adam(vanilla_model.parameters(), lr=LR)
    
    for ep in range(1, EPOCHS_STAGE1 + 1):
        stats = train_one_epoch_stage1(
            model=vanilla_model,
            optimizer=vanilla_opt,
            sentence_stream_fn=sentence_stream_factory,
            batches_per_epoch=BATCHES_PER_EPOCH,
            batch_size=BATCH_SIZE,
            window_size=WINDOW_SIZE,
            neg_samples=NEG_SAMPLES,
            neg_dist=neg_dist,
            word2idx=word2idx,
            keep_prob=keep_prob,
            rsr_weight=0.0,  # No RSR for vanilla
            pairs_array=None,
        )
        print(f"  [vanilla] epoch {ep}/{EPOCHS_STAGE1} | loss={stats['loss']:.4f}")
    
    # =========================================================================
    # Train RSR with Distillation (two stages)
    # =========================================================================
    print(f"[seed={seed}] Training RSR (Stage 1: W2V + RSR)...")
    rsr_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    rsr_model.load_state_dict(deepcopy(init_state))
    rsr_opt = optim.Adam(rsr_model.parameters(), lr=LR)
    
    # STAGE 1: Standard RSR training
    for ep in range(1, EPOCHS_STAGE1 + 1):
        stats = train_one_epoch_stage1(
            model=rsr_model,
            optimizer=rsr_opt,
            sentence_stream_fn=sentence_stream_factory,
            batches_per_epoch=BATCHES_PER_EPOCH,
            batch_size=BATCH_SIZE,
            window_size=WINDOW_SIZE,
            neg_samples=NEG_SAMPLES,
            neg_dist=neg_dist,
            word2idx=word2idx,
            keep_prob=keep_prob,
            rsr_weight=RSR_WEIGHT,
            rsr_pairs_per_step=RSR_PAIRS_PER_STEP,
            soft_rank_strength=SOFT_RANK_STRENGTH,
            pairs_array=pairs_array,
        )
        print(f"  [rsr-stage1] epoch {ep}/{EPOCHS_STAGE1} | w2v={stats['w2v']:.4f} | rsr={stats['rsr']:.4f}")
    
    # STAGE 2: Distillation to unseen words
    print(f"[seed={seed}] Training RSR (Stage 2: Distillation)...")
    distill_opt = optim.Adam(rsr_model.parameters(), lr=DISTILL_LR)
    
    for ep in range(1, EPOCHS_STAGE2 + 1):
        stats = train_stage2_distillation(
            model=rsr_model,
            optimizer=distill_opt,
            trained_indices_list=trained_indices_list,
            vocab_size=vocab_size,
            distill_batches=DISTILL_BATCHES,
            distill_batch_size=DISTILL_BATCH_SIZE,
            k_neighbors=DISTILL_K_NEIGHBORS,
            n_targets=DISTILL_N_TARGETS,
        )
        print(f"  [rsr-stage2] epoch {ep}/{EPOCHS_STAGE2} | distill_loss={stats['loss']:.4f}")
    
    # =========================================================================
    # Evaluate
    # =========================================================================
    print(f"[seed={seed}] Evaluating...")
    
    v_all = simlex_spearman_by_train_count(vanilla_model, simlex_df, word2idx, train_words, train_count=None)
    r_all = simlex_spearman_by_train_count(rsr_model, simlex_df, word2idx, train_words, train_count=None)
    
    v_0 = simlex_spearman_by_train_count(vanilla_model, simlex_df, word2idx, train_words, train_count=0)
    r_0 = simlex_spearman_by_train_count(rsr_model, simlex_df, word2idx, train_words, train_count=0)
    
    v_1 = simlex_spearman_by_train_count(vanilla_model, simlex_df, word2idx, train_words, train_count=1)
    r_1 = simlex_spearman_by_train_count(rsr_model, simlex_df, word2idx, train_words, train_count=1)
    
    v_2 = simlex_spearman_by_train_count(vanilla_model, simlex_df, word2idx, train_words, train_count=2)
    r_2 = simlex_spearman_by_train_count(rsr_model, simlex_df, word2idx, train_words, train_count=2)
    
    print(f"  SimLex (all):      vanilla={v_all['rho']:.4f}  rsr={r_all['rho']:.4f}  n={v_all['n']}")
    print(f"  SimLex (0 train):  vanilla={v_0['rho']:.4f}  rsr={r_0['rho']:.4f}  n={v_0['n']}")
    print(f"  SimLex (1 train):  vanilla={v_1['rho']:.4f}  rsr={r_1['rho']:.4f}  n={v_1['n']}")
    print(f"  SimLex (2 train):  vanilla={v_2['rho']:.4f}  rsr={r_2['rho']:.4f}  n={v_2['n']}")
    
    delta_0 = r_0['rho'] - v_0['rho']
    print(f"\n  *** DELTA 0-TRAIN: {delta_0:+.4f} ***")
    
    del vanilla_model, rsr_model, init_state
    torch.cuda.empty_cache()
    
    return {
        "seed": seed,
        "vanilla_all": v_all["rho"],
        "rsr_all": r_all["rho"],
        "delta_all": r_all["rho"] - v_all["rho"],
        "n_all": v_all["n"],
        "vanilla_0_train": v_0["rho"],
        "rsr_0_train": r_0["rho"],
        "delta_0_train": delta_0,
        "n_0_train": v_0["n"],
        "vanilla_1_train": v_1["rho"],
        "rsr_1_train": r_1["rho"],
        "delta_1_train": r_1["rho"] - v_1["rho"],
        "n_1_train": v_1["n"],
        "vanilla_2_train": v_2["rho"],
        "rsr_2_train": r_2["rho"],
        "delta_2_train": r_2["rho"] - v_2["rho"],
        "n_2_train": v_2["n"],
    }

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RSR with Similarity Distillation")
    parser.add_argument("--start_seed", type=int, default=1, help="Starting seed")
    parser.add_argument("--end_seed", type=int, default=20, help="Ending seed")
    parser.add_argument("--output", type=str, default="results/seed_results_distill.xlsx", help="Output Excel file")
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Running seeds {args.start_seed} to {args.end_seed}")
    print(f"Output: {args.output}")
    print(f"\n*** TWO-STAGE RSR WITH DISTILLATION (VECTORIZED) ***")
    print(f"  Stage 1: {EPOCHS_STAGE1} epochs of W2V + RSR (trained words)")
    print(f"  Stage 2: {EPOCHS_STAGE2} epochs of distillation (unseen words)")
    print(f"  Distillation params: K={DISTILL_K_NEIGHBORS} neighbors, {DISTILL_N_TARGETS} shared targets, {DISTILL_BATCHES} batches")
    
    # Build vocab
    print("\n" + "="*70)
    print("Building vocabulary (one-time)")
    print("="*70)
    
    def sentence_stream_factory():
        return iter_wiki_sentences_jsonl(WIKI_DIR, max_files=MAX_JSON_FILES, max_articles=MAX_ARTICLES)
    
    word2idx, idx2word, idx_counts = build_vocab_from_stream(
        sentence_stream_factory(),
        min_count=MIN_COUNT,
        max_vocab=MAX_VOCAB,
    )
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size:,}")
    
    # Load datasets - now also returns trained indices
    train_words, pairs_array, trained_indices = load_all_datasets(word2idx)
    print(f"Trained word indices: {len(trained_indices):,}")
    
    # Load SimLex
    print("\nLoading SimLex-999 for evaluation...")
    simlex_df = load_simlex(SIMLEX_PATH)
    print(f"SimLex pairs: {len(simlex_df):,}")
    
    simlex_words = set(simlex_df["word1"].tolist()) | set(simlex_df["word2"].tolist())
    overlap = simlex_words & train_words
    print(f"SimLex words in training vocab: {len(overlap):,} / {len(simlex_words):,}")
    
    # Resume support
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        existing_df = pd.read_excel(output_path)
        existing_seeds = set(existing_df["seed"].tolist())
        results = existing_df.to_dict("records")
        print(f"\nResuming from existing file. Already completed seeds: {sorted(existing_seeds)}")
    else:
        existing_seeds = set()
        results = []
    
    # Run seeds
    for seed in range(args.start_seed, args.end_seed + 1):
        if seed in existing_seeds:
            print(f"\n[SKIP] Seed {seed} already completed")
            continue
        
        result = run_one_seed(seed, word2idx, idx2word, idx_counts, train_words, pairs_array, 
                              trained_indices, simlex_df)
        results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values("seed")
        df.to_excel(output_path, index=False)
        print(f"[SAVED] {output_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results)
    df = df.sort_values("seed")
    
    summary_cols = ["seed", "vanilla_0_train", "rsr_0_train", "delta_0_train", 
                    "vanilla_1_train", "rsr_1_train", "vanilla_2_train", "rsr_2_train"]
    print(df[summary_cols].to_string(index=False))
    
    print(f"\n--- Mean Deltas (RSR - Vanilla) ---")
    print(f"  All pairs:      {df['delta_all'].mean():.4f} ± {df['delta_all'].std():.4f}")
    print(f"  0 train words:  {df['delta_0_train'].mean():.4f} ± {df['delta_0_train'].std():.4f}  <-- KEY METRIC!")
    print(f"  1 train word:   {df['delta_1_train'].mean():.4f} ± {df['delta_1_train'].std():.4f}")
    print(f"  2 train words:  {df['delta_2_train'].mean():.4f} ± {df['delta_2_train'].std():.4f}")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()

