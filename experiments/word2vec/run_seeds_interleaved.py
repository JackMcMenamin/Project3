"""
Multi-seed training with INTERLEAVED RSR training.

Instead of combining losses with weights, we ALTERNATE between:
  - W2V steps: Pure skip-gram training on full vocabulary
  - RSR steps: Pure similarity alignment on training pairs

Theory: W2V pulls co-occurring words together. If "cat" gets RSR training
and "dog" doesn't, W2V will still pull dog toward cat (they co-occur).
So dog indirectly benefits from cat's RSR training.

Uses full Wikipedia corpus (74GB) for strong W2V signal.

This version sweeps across RSR_FREQUENCY values (1-20) and seeds (1-10),
producing 200 total rows to find the optimal RSR frequency.

Usage:
    python run_seeds_interleaved.py
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

# Repo root is three levels up: experiments/word2vec/run_seeds_interleaved.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = REPO_ROOT / "data"
WIKI_DIR = BASE_DATA_DIR / "enwiki_namespace_0"

# RSR training datasets
MEN_PATH = BASE_DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"
SIMVERB_PATH = BASE_DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
THINGS_DIR = REPO_ROOT / "things_similarity"
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"

# SimLex-999 for evaluation (consolidated under data/ in the restructure)
SIMLEX_PATH = BASE_DATA_DIR / "SimLex-999" / "SimLex-999.txt"

MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = REPO_ROOT / "results"
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

# INTERLEAVED TRAINING CONFIG
# RSR_FREQUENCY = N means RSR every Nth step
# RSR_FREQUENCY = 10 -> 90% W2V, 10% RSR
# RSR_FREQUENCY = 5  -> 80% W2V, 20% RSR
# We sweep RSR_FREQUENCY from 1 to 20
RSR_FREQUENCY_MIN = 1
RSR_FREQUENCY_MAX = 20
SEEDS_PER_FREQUENCY = 10  # Seeds 1-10 for each frequency
RSR_PAIRS_PER_STEP = 5000
SOFT_RANK_STRENGTH = 2.0

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
    
    return all_words, combined

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
# Training (Interleaved)
# ==============================================================================

def cosine_sim_from_in_embeddings_grad(model, idx_a, idx_b):
    va = model.in_embed(idx_a)
    vb = model.in_embed(idx_b)
    va = va / (va.norm(dim=1, keepdim=True) + 1e-8)
    vb = vb / (vb.norm(dim=1, keepdim=True) + 1e-8)
    return (va * vb).sum(dim=1)

def train_one_epoch_interleaved(
    model, optimizer, sentence_stream_fn, batches_per_epoch, batch_size,
    window_size, neg_samples, neg_dist, word2idx, keep_prob,
    rsr_frequency=10, rsr_pairs_per_step=5000, soft_rank_strength=2.0,
    pairs_array=None
):
    """
    Interleaved training: alternate between W2V and RSR steps.
    
    - Every rsr_frequency-th step: PURE RSR (no W2V loss)
    - All other steps: PURE W2V (no RSR loss)
    
    This is different from weighted combination - each step has ONE loss only.
    """
    model.train()

    pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size, word2idx, keep_prob)
    batch_iter = batch_pairs(pair_iter, batch_size)

    total_loss = 0.0
    total_w2v = 0.0
    total_rsr = 0.0
    w2v_steps = 0
    rsr_steps = 0

    pbar = tqdm(range(batches_per_epoch), desc="training", leave=False)
    for b in pbar:
        
        if b % rsr_frequency == 0 and pairs_array is not None:
            # =================================================================
            # RSR STEP: Pure similarity alignment (no W2V)
            # =================================================================
            i_idx, j_idx, target_sim = sample_rsr_pairs(rsr_pairs_per_step, pairs_array)
            pred_sim = cosine_sim_from_in_embeddings_grad(model, i_idx, j_idx)
            rho = soft_spearman(pred_sim, target_sim, regularization_strength=soft_rank_strength)
            loss = 1.0 - rho
            
            total_rsr += float(loss.item())
            rsr_steps += 1
            
        else:
            # =================================================================
            # W2V STEP: Pure skip-gram (no RSR)
            # =================================================================
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
            loss = w2v_neg_sampling_loss(pos_logits, neg_logits)
            
            total_w2v += float(loss.item())
            w2v_steps += 1

        # Single loss backprop (NOT combined!)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return {
        "loss": total_loss / batches_per_epoch,
        "w2v": total_w2v / max(1, w2v_steps),
        "rsr": total_rsr / max(1, rsr_steps),
        "w2v_steps": w2v_steps,
        "rsr_steps": rsr_steps,
    }

def train_one_epoch_vanilla(
    model, optimizer, sentence_stream_fn, batches_per_epoch, batch_size,
    window_size, neg_samples, neg_dist, word2idx, keep_prob
):
    """Pure W2V training (no RSR) for vanilla baseline."""
    model.train()

    pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size, word2idx, keep_prob)
    batch_iter = batch_pairs(pair_iter, batch_size)

    total_loss = 0.0

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
        loss = w2v_neg_sampling_loss(pos_logits, neg_logits)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return {"loss": total_loss / batches_per_epoch}

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

def run_one_seed(seed: int, rsr_frequency: int, word2idx, idx2word, idx_counts, train_words, pairs_array, simlex_df):
    print(f"\n{'='*70}")
    print(f"SEED {seed} | RSR_FREQUENCY {rsr_frequency}")
    print(f"{'='*70}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    vocab_size = len(word2idx)
    
    neg_dist = make_unigram_dist(idx_counts)
    keep_prob = make_subsampling_keep_probs(idx_counts, t=SUBSAMPLE_T) if SUBSAMPLE_T else None
    
    def sentence_stream_factory():
        return iter_wiki_sentences_jsonl(WIKI_DIR, max_files=MAX_JSON_FILES, max_articles=MAX_ARTICLES)
    
    # Shared init
    init_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    init_state = deepcopy(init_model.state_dict())
    del init_model
    
    # =========================================================================
    # Train Vanilla (pure W2V)
    # =========================================================================
    print(f"[seed={seed}] Training Vanilla (pure W2V)...")
    vanilla_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    vanilla_model.load_state_dict(deepcopy(init_state))
    vanilla_opt = optim.Adam(vanilla_model.parameters(), lr=LR)
    
    for ep in range(1, EPOCHS + 1):
        stats = train_one_epoch_vanilla(
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
        )
        print(f"  [vanilla] epoch {ep}/{EPOCHS} | loss={stats['loss']:.4f}")
    
    # =========================================================================
    # Train RSR (interleaved W2V + RSR)
    # =========================================================================
    print(f"[seed={seed}] Training RSR (interleaved, RSR_FREQUENCY={rsr_frequency})...")
    rsr_model = SkipGramWord2Vec(vocab_size, EMBEDDING_DIM).to(DEVICE)
    rsr_model.load_state_dict(deepcopy(init_state))
    rsr_opt = optim.Adam(rsr_model.parameters(), lr=LR)
    
    for ep in range(1, EPOCHS + 1):
        stats = train_one_epoch_interleaved(
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
            rsr_frequency=rsr_frequency,
            rsr_pairs_per_step=RSR_PAIRS_PER_STEP,
            soft_rank_strength=SOFT_RANK_STRENGTH,
            pairs_array=pairs_array,
        )
        print(f"  [interleaved] epoch {ep}/{EPOCHS} | w2v={stats['w2v']:.4f} | rsr={stats['rsr']:.4f} | steps: {stats['w2v_steps']}W/{stats['rsr_steps']}R")
    
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
    
    del vanilla_model, rsr_model, init_state
    torch.cuda.empty_cache()
    
    return {
        "rsr_frequency": rsr_frequency,
        "seed": seed,
        "vanilla_all": v_all["rho"],
        "rsr_all": r_all["rho"],
        "delta_all": r_all["rho"] - v_all["rho"],
        "n_all": v_all["n"],
        "vanilla_0_train": v_0["rho"],
        "rsr_0_train": r_0["rho"],
        "delta_0_train": r_0["rho"] - v_0["rho"],
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
    parser = argparse.ArgumentParser(description="Interleaved RSR Training - Sweep RSR_FREQUENCY")
    parser.add_argument("--output", type=str, default="results/seed_results_interleaved_sweep.xlsx", help="Output Excel file")
    args = parser.parse_args()
    
    total_runs = (RSR_FREQUENCY_MAX - RSR_FREQUENCY_MIN + 1) * SEEDS_PER_FREQUENCY
    
    print(f"Device: {DEVICE}")
    print(f"\n*** INTERLEAVED TRAINING - FREQUENCY SWEEP ***")
    print(f"  RSR_FREQUENCY range: {RSR_FREQUENCY_MIN} to {RSR_FREQUENCY_MAX}")
    print(f"  Seeds per frequency: 1 to {SEEDS_PER_FREQUENCY}")
    print(f"  Total runs: {total_runs}")
    print(f"  RSR_PAIRS_PER_STEP: {RSR_PAIRS_PER_STEP}")
    print(f"Output: {args.output}")
    
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
    
    # Load datasets
    train_words, pairs_array = load_all_datasets(word2idx)
    
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
        # Track completed (frequency, seed) pairs
        existing_pairs = set(zip(existing_df["rsr_frequency"].tolist(), existing_df["seed"].tolist()))
        results = existing_df.to_dict("records")
        print(f"\nResuming from existing file. Already completed: {len(existing_pairs)} runs")
    else:
        existing_pairs = set()
        results = []
    
    # Run sweeps: outer loop = frequency, inner loop = seeds
    run_count = 0
    for rsr_freq in range(RSR_FREQUENCY_MIN, RSR_FREQUENCY_MAX + 1):
        for seed in range(1, SEEDS_PER_FREQUENCY + 1):
            if (rsr_freq, seed) in existing_pairs:
                print(f"\n[SKIP] RSR_FREQUENCY={rsr_freq}, Seed={seed} already completed")
                continue
            
            run_count += 1
            remaining = total_runs - len(existing_pairs) - run_count + 1
            print(f"\n>>> Run {run_count}/{remaining + run_count - 1} remaining <<<")
            
            result = run_one_seed(seed, rsr_freq, word2idx, idx2word, idx_counts, train_words, pairs_array, simlex_df)
            results.append(result)
            
            df = pd.DataFrame(results)
            df = df.sort_values(["rsr_frequency", "seed"])
            df.to_excel(output_path, index=False)
            print(f"[SAVED] {output_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results)
    df = df.sort_values(["rsr_frequency", "seed"])
    
    # Summary by RSR_FREQUENCY
    print("\n--- Mean Delta (0-train) by RSR_FREQUENCY ---")
    print(f"{'Freq':>6} | {'Mean Δ0':>10} | {'Std Δ0':>10} | {'Mean Δ2':>10} | {'Seeds':>6}")
    print("-" * 55)
    
    for freq in range(RSR_FREQUENCY_MIN, RSR_FREQUENCY_MAX + 1):
        freq_df = df[df["rsr_frequency"] == freq]
        if len(freq_df) == 0:
            continue
        mean_d0 = freq_df["delta_0_train"].mean()
        std_d0 = freq_df["delta_0_train"].std()
        mean_d2 = freq_df["delta_2_train"].mean()
        n_seeds = len(freq_df)
        highlight = "  <-- BEST" if mean_d0 == df.groupby("rsr_frequency")["delta_0_train"].mean().max() else ""
        print(f"{freq:>6} | {mean_d0:>10.4f} | {std_d0:>10.4f} | {mean_d2:>10.4f} | {n_seeds:>6}{highlight}")
    
    # Overall stats
    print(f"\n--- Overall Mean Deltas (RSR - Vanilla) ---")
    print(f"  All pairs:      {df['delta_all'].mean():.4f} ± {df['delta_all'].std():.4f}")
    print(f"  0 train words:  {df['delta_0_train'].mean():.4f} ± {df['delta_0_train'].std():.4f}  <-- KEY METRIC!")
    print(f"  1 train word:   {df['delta_1_train'].mean():.4f} ± {df['delta_1_train'].std():.4f}")
    print(f"  2 train words:  {df['delta_2_train'].mean():.4f} ± {df['delta_2_train'].std():.4f}")
    
    # Find best frequency
    best_freq = df.groupby("rsr_frequency")["delta_0_train"].mean().idxmax()
    best_delta = df.groupby("rsr_frequency")["delta_0_train"].mean().max()
    print(f"\n*** BEST RSR_FREQUENCY for 0-train: {best_freq} (mean delta = {best_delta:.4f}) ***")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()

