"""
Multi-seed training loop for Vanilla vs RSR Word2Vec comparison.
Uses MEN dataset for RSR training (instead of THINGS).
Evaluates on SimLex-999 with breakdown by MEN membership.

Usage:
    python run_seeds_men.py
    python run_seeds_men.py --start_seed 1 --end_seed 20
    python run_seeds_men.py --start_seed 5 --end_seed 10  # resume from seed 5
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

from scipy.stats import spearmanr

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DATA_DIR = Path("data")
WIKI_DIR = BASE_DATA_DIR / "enwiki_namespace_0"

# MEN dataset for RSR training
MEN_PATH = BASE_DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"

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

EPOCHS = 2
BATCH_SIZE = 8192
LR = 2e-3
BATCHES_PER_EPOCH = 10000

RSR_WEIGHT = 0.01
RSR_EVERY_N_BATCHES = 5
RSR_PAIRS_PER_STEP = 2000  # MEN has 3000 pairs, so sample fewer per step
SOFT_RANK_STRENGTH = 2.0

RSR_WARMUP_FRAC = 0.6
RSR_RAMP_FRAC = 0.2
RSR_RAMP_CHUNKS = 10

MAX_JSON_FILES = None
MAX_ARTICLES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Tokenization + streaming
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
# MEN loader (for RSR training)
# ==============================================================================

def load_men_for_rsr(path: Path, word2idx: dict):
    """
    Load MEN dataset and return pairs that are in our vocabulary.
    
    MEN format: "word1-POS word2-POS score" (e.g., "sun-n sunlight-n 50.0")
    We strip the POS tags.
    
    Returns:
        men_words: set of unique words in MEN (that are in vocab)
        men_pairs: list of (vocab_idx_1, vocab_idx_2, similarity_score)
    """
    if not path.exists():
        raise FileNotFoundError(f"MEN file not found: {path}")
    
    men_words = set()
    men_pairs = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            # Strip POS tags (e.g., "sun-n" -> "sun")
            w1 = parts[0].rsplit("-", 1)[0].lower()
            w2 = parts[1].rsplit("-", 1)[0].lower()
            
            try:
                score = float(parts[2])
            except:
                continue
            
            # Check if both words are in vocabulary
            idx1 = word2idx.get(w1, 0)
            idx2 = word2idx.get(w2, 0)
            
            if idx1 != 0 and idx2 != 0:
                men_words.add(w1)
                men_words.add(w2)
                men_pairs.append((idx1, idx2, score))
    
    # Convert to numpy arrays for efficient sampling
    men_pairs_array = np.array(men_pairs, dtype=np.float32)
    
    # Normalize scores to [0, 1] for consistency
    if len(men_pairs_array) > 0:
        min_score = men_pairs_array[:, 2].min()
        max_score = men_pairs_array[:, 2].max()
        if max_score > min_score:
            men_pairs_array[:, 2] = (men_pairs_array[:, 2] - min_score) / (max_score - min_score)
    
    return men_words, men_pairs_array

def sample_men_pairs(num_pairs: int, men_pairs_array: np.ndarray):
    """
    Sample pairs from MEN dataset for RSR training.
    
    Returns:
        vocab_i: (P,) tensor of vocab indices
        vocab_j: (P,) tensor of vocab indices
        target_sim: (P,) tensor of similarity scores
    """
    n_available = len(men_pairs_array)
    if num_pairs > n_available:
        # Sample with replacement if we need more pairs than available
        idx = np.random.randint(0, n_available, size=num_pairs)
    else:
        # Sample without replacement
        idx = np.random.choice(n_available, size=num_pairs, replace=False)
    
    sampled = men_pairs_array[idx]
    
    return (
        torch.tensor(sampled[:, 0].astype(np.int64), dtype=torch.long, device=DEVICE),
        torch.tensor(sampled[:, 1].astype(np.int64), dtype=torch.long, device=DEVICE),
        torch.tensor(sampled[:, 2], dtype=torch.float32, device=DEVICE),
    )

# ==============================================================================
# Training
# ==============================================================================

def cosine_sim_from_in_embeddings_grad(model, idx_a, idx_b):
    va = model.in_embed(idx_a)
    vb = model.in_embed(idx_b)
    va = va / (va.norm(dim=1, keepdim=True) + 1e-8)
    vb = vb / (vb.norm(dim=1, keepdim=True) + 1e-8)
    return (va * vb).sum(dim=1)

def train_one_epoch_streaming(
    model, optimizer, sentence_stream_fn, batches_per_epoch, batch_size,
    window_size, neg_samples, neg_dist, word2idx, keep_prob,
    rsr_weight=0.0, rsr_every_n=10, rsr_pairs_per_step=2000,
    soft_rank_strength=2.0, men_pairs_array=None
):
    model.train()

    pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size, word2idx, keep_prob)
    batch_iter = batch_pairs(pair_iter, batch_size)

    total_loss = 0.0
    total_w2v = 0.0
    total_rsr = 0.0

    pbar = tqdm(range(batches_per_epoch), desc="training", leave=False)
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
        loss_w2v = w2v_neg_sampling_loss(pos_logits, neg_logits)

        loss_rsr = torch.tensor(0.0, device=DEVICE)
        if rsr_weight > 0.0 and men_pairs_array is not None and (b % rsr_every_n == 0):
            i_idx, j_idx, target_sim = sample_men_pairs(rsr_pairs_per_step, men_pairs_array)
            pred_sim = cosine_sim_from_in_embeddings_grad(model, i_idx, j_idx)
            rho = soft_spearman(pred_sim, target_sim, regularization_strength=soft_rank_strength)
            loss_rsr = 1.0 - rho

        loss = loss_w2v + (rsr_weight * loss_rsr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_w2v += float(loss_w2v.item())
        total_rsr += float(loss_rsr.item()) if rsr_weight > 0.0 else 0.0

    return {
        "loss": total_loss / batches_per_epoch,
        "w2v": total_w2v / batches_per_epoch,
        "rsr": total_rsr / max(1, (batches_per_epoch // rsr_every_n)) if rsr_weight > 0.0 else 0.0,
    }

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
def simlex_spearman_by_men_count(model, simlex_df, word2idx, men_words, men_count=None):
    """
    Evaluate SimLex-999 Spearman correlation, optionally filtered by MEN membership.
    
    Args:
        men_count: None = all pairs, 0 = neither word in MEN, 
                   1 = exactly one word in MEN, 2 = both words in MEN
    """
    model.eval()
    W = model.in_embed.weight.detach()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    sims = []
    scores = []
    covered = 0

    for _, row in simlex_df.iterrows():
        w1 = row["word1"]
        w2 = row["word2"]

        # Count how many words are in MEN
        w1_in_men = w1 in men_words
        w2_in_men = w2 in men_words
        pair_men_count = int(w1_in_men) + int(w2_in_men)

        # Filter by men_count if specified
        if men_count is not None and pair_men_count != men_count:
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

def run_one_seed(seed: int, word2idx, idx2word, idx_counts, men_words, men_pairs_array, simlex_df):
    """Run vanilla + RSR training for one seed, return results dict."""
    
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    vocab_size = len(word2idx)
    
    # Distributions
    neg_dist = make_unigram_dist(idx_counts)
    keep_prob = make_subsampling_keep_probs(idx_counts, t=SUBSAMPLE_T) if SUBSAMPLE_T else None
    
    # Sentence stream factory
    def sentence_stream_factory():
        return iter_wiki_sentences_jsonl(WIKI_DIR, max_files=MAX_JSON_FILES, max_articles=MAX_ARTICLES)
    
    # Shared init
    init_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    init_state = deepcopy(init_model.state_dict())
    del init_model
    
    # =========================================================================
    # Train Vanilla
    # =========================================================================
    print(f"[seed={seed}] Training Vanilla...")
    vanilla_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    vanilla_model.load_state_dict(deepcopy(init_state))
    vanilla_opt = optim.Adam(vanilla_model.parameters(), lr=LR)
    
    for ep in range(1, EPOCHS + 1):
        stats = train_one_epoch_streaming(
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
            rsr_weight=0.0,
            men_pairs_array=None,
        )
        print(f"  [vanilla] epoch {ep}/{EPOCHS} | loss={stats['loss']:.4f}")
    
    # =========================================================================
    # Train RSR (with ramp-up, using MEN)
    # =========================================================================
    print(f"[seed={seed}] Training RSR (MEN)...")
    rsr_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    rsr_model.load_state_dict(deepcopy(init_state))
    rsr_opt = optim.Adam(rsr_model.parameters(), lr=LR)
    
    sentence_iter = sentence_stream_factory()
    def _same_stream_forever():
        return sentence_iter
    
    warmup_batches = int(BATCHES_PER_EPOCH * RSR_WARMUP_FRAC)
    ramp_batches_total = int(BATCHES_PER_EPOCH * RSR_RAMP_FRAC)
    steady_batches = BATCHES_PER_EPOCH - warmup_batches - ramp_batches_total
    
    if ramp_batches_total > 0:
        base = ramp_batches_total // RSR_RAMP_CHUNKS
        rem = ramp_batches_total % RSR_RAMP_CHUNKS
        ramp_chunk_sizes = [base + (1 if i < rem else 0) for i in range(RSR_RAMP_CHUNKS)]
        ramp_chunk_sizes = [b for b in ramp_chunk_sizes if b > 0]
    else:
        ramp_chunk_sizes = []
    
    for ep in range(1, EPOCHS + 1):
        # Warmup
        if warmup_batches > 0:
            train_one_epoch_streaming(
                model=rsr_model, optimizer=rsr_opt,
                sentence_stream_fn=_same_stream_forever,
                batches_per_epoch=warmup_batches,
                batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
                neg_samples=NEG_SAMPLES, neg_dist=neg_dist,
                word2idx=word2idx, keep_prob=keep_prob,
                rsr_weight=0.0, men_pairs_array=None,
            )
        
        # Ramp
        for i, chunk_batches in enumerate(ramp_chunk_sizes, start=1):
            t = i / float(len(ramp_chunk_sizes))
            w = RSR_WEIGHT * t
            train_one_epoch_streaming(
                model=rsr_model, optimizer=rsr_opt,
                sentence_stream_fn=_same_stream_forever,
                batches_per_epoch=chunk_batches,
                batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
                neg_samples=NEG_SAMPLES, neg_dist=neg_dist,
                word2idx=word2idx, keep_prob=keep_prob,
                rsr_weight=w,
                rsr_every_n=RSR_EVERY_N_BATCHES,
                rsr_pairs_per_step=RSR_PAIRS_PER_STEP,
                soft_rank_strength=SOFT_RANK_STRENGTH,
                men_pairs_array=men_pairs_array,
            )
        
        # Steady
        if steady_batches > 0:
            stats = train_one_epoch_streaming(
                model=rsr_model, optimizer=rsr_opt,
                sentence_stream_fn=_same_stream_forever,
                batches_per_epoch=steady_batches,
                batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
                neg_samples=NEG_SAMPLES, neg_dist=neg_dist,
                word2idx=word2idx, keep_prob=keep_prob,
                rsr_weight=RSR_WEIGHT,
                rsr_every_n=RSR_EVERY_N_BATCHES,
                rsr_pairs_per_step=RSR_PAIRS_PER_STEP,
                soft_rank_strength=SOFT_RANK_STRENGTH,
                men_pairs_array=men_pairs_array,
            )
        print(f"  [rsr] epoch {ep}/{EPOCHS} | loss={stats['loss']:.4f}")
    
    # =========================================================================
    # Evaluate on SimLex-999 (by MEN membership count)
    # =========================================================================
    print(f"[seed={seed}] Evaluating...")
    
    # All pairs (for reference)
    v_all = simlex_spearman_by_men_count(vanilla_model, simlex_df, word2idx, men_words, men_count=None)
    r_all = simlex_spearman_by_men_count(rsr_model, simlex_df, word2idx, men_words, men_count=None)
    
    # 0 MEN words (neither word in MEN)
    v_0 = simlex_spearman_by_men_count(vanilla_model, simlex_df, word2idx, men_words, men_count=0)
    r_0 = simlex_spearman_by_men_count(rsr_model, simlex_df, word2idx, men_words, men_count=0)
    
    # 1 MEN word (exactly one word in MEN)
    v_1 = simlex_spearman_by_men_count(vanilla_model, simlex_df, word2idx, men_words, men_count=1)
    r_1 = simlex_spearman_by_men_count(rsr_model, simlex_df, word2idx, men_words, men_count=1)
    
    # 2 MEN words (both words in MEN)
    v_2 = simlex_spearman_by_men_count(vanilla_model, simlex_df, word2idx, men_words, men_count=2)
    r_2 = simlex_spearman_by_men_count(rsr_model, simlex_df, word2idx, men_words, men_count=2)
    
    print(f"  SimLex (all):      vanilla={v_all['rho']:.4f}  rsr={r_all['rho']:.4f}  n={v_all['n']}")
    print(f"  SimLex (0 MEN):    vanilla={v_0['rho']:.4f}  rsr={r_0['rho']:.4f}  n={v_0['n']}")
    print(f"  SimLex (1 MEN):    vanilla={v_1['rho']:.4f}  rsr={r_1['rho']:.4f}  n={v_1['n']}")
    print(f"  SimLex (2 MEN):    vanilla={v_2['rho']:.4f}  rsr={r_2['rho']:.4f}  n={v_2['n']}")
    
    # Cleanup
    del vanilla_model, rsr_model, init_state
    torch.cuda.empty_cache()
    
    return {
        "seed": seed,
        # All pairs
        "vanilla_all": v_all["rho"],
        "rsr_all": r_all["rho"],
        "delta_all": r_all["rho"] - v_all["rho"],
        "n_all": v_all["n"],
        # 0 MEN (neither word)
        "vanilla_0_men": v_0["rho"],
        "rsr_0_men": r_0["rho"],
        "delta_0_men": r_0["rho"] - v_0["rho"],
        "n_0_men": v_0["n"],
        # 1 MEN (one word)
        "vanilla_1_men": v_1["rho"],
        "rsr_1_men": r_1["rho"],
        "delta_1_men": r_1["rho"] - v_1["rho"],
        "n_1_men": v_1["n"],
        # 2 MEN (both words)
        "vanilla_2_men": v_2["rho"],
        "rsr_2_men": r_2["rho"],
        "delta_2_men": r_2["rho"] - v_2["rho"],
        "n_2_men": v_2["n"],
    }

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-seed Vanilla vs RSR (MEN) training")
    parser.add_argument("--start_seed", type=int, default=1, help="Starting seed (default: 1)")
    parser.add_argument("--end_seed", type=int, default=20, help="Ending seed (default: 20)")
    parser.add_argument("--output", type=str, default="results/seed_results_men.xlsx", help="Output Excel file")
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Running seeds {args.start_seed} to {args.end_seed}")
    print(f"Output: {args.output}")
    print(f"RSR training data: MEN ({MEN_PATH})")
    
    # =========================================================================
    # Build vocab ONCE (this is the slow part, ~15 min)
    # =========================================================================
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
    
    # =========================================================================
    # Load MEN data ONCE
    # =========================================================================
    print("\nLoading MEN dataset for RSR training...")
    men_words, men_pairs_array = load_men_for_rsr(MEN_PATH, word2idx)
    print(f"MEN words in vocab: {len(men_words):,}")
    print(f"MEN pairs in vocab: {len(men_pairs_array):,}")
    
    # =========================================================================
    # Load SimLex ONCE
    # =========================================================================
    print("\nLoading SimLex-999 for evaluation...")
    simlex_df = load_simlex(SIMLEX_PATH)
    print(f"SimLex pairs: {len(simlex_df):,}")
    
    # =========================================================================
    # Load existing results if any (for resume)
    # =========================================================================
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
    
    # =========================================================================
    # Run seeds
    # =========================================================================
    for seed in range(args.start_seed, args.end_seed + 1):
        if seed in existing_seeds:
            print(f"\n[SKIP] Seed {seed} already completed")
            continue
        
        result = run_one_seed(seed, word2idx, idx2word, idx_counts, men_words, men_pairs_array, simlex_df)
        results.append(result)
        
        # Save after each seed (incremental save)
        df = pd.DataFrame(results)
        df = df.sort_values("seed")
        df.to_excel(output_path, index=False)
        print(f"[SAVED] {output_path}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results)
    df = df.sort_values("seed")
    
    # Print a compact view of key columns
    summary_cols = ["seed", "vanilla_0_men", "rsr_0_men", "vanilla_1_men", "rsr_1_men", "vanilla_2_men", "rsr_2_men"]
    print(df[summary_cols].to_string(index=False))
    
    print(f"\n--- Mean Deltas (RSR - Vanilla) ---")
    print(f"  All pairs:    {df['delta_all'].mean():.4f} ± {df['delta_all'].std():.4f}  (n={df['n_all'].iloc[0]})")
    print(f"  0 MEN words:  {df['delta_0_men'].mean():.4f} ± {df['delta_0_men'].std():.4f}  (n={df['n_0_men'].iloc[0]})")
    print(f"  1 MEN word:   {df['delta_1_men'].mean():.4f} ± {df['delta_1_men'].std():.4f}  (n={df['n_1_men'].iloc[0]})")
    print(f"  2 MEN words:  {df['delta_2_men'].mean():.4f} ± {df['delta_2_men'].std():.4f}  (n={df['n_2_men'].iloc[0]})")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()

