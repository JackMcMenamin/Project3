# ==============================================================================
# Word2Vec-from-scratch (Vanilla vs RSR) + SimLex-999 evaluation
# - Model A: train from scratch on Wikipedia (skip-gram + negative sampling)
# - Model B: train from scratch on Wikipedia + THINGS-informed RSR loss
# - Evaluate both on SimLex-999 (Spearman vs human similarity)
#
# NOTE: This script is written "notebook-style" with #-------------- separators.
# ==============================================================================

import os
import re
import json
import math
import random
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
# Repro + device
# ==============================================================================
SEED = 420

def set_seed(seed: int = 420):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] SEED={SEED} | DEVICE={DEVICE}")

#------------------------------------------------------------------------------
# Paths + configuration
#------------------------------------------------------------------------------

# Repo root is three levels up: experiments/word2vec/main.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = REPO_ROOT / "data"

# Wikipedia corpus directory with JSON files (each file is an array of articles with "text")
WIKI_DIR = BASE_DATA_DIR / "enwiki20201020"     # <--- change if yours differs

# THINGS similarity bundle
THINGS_DIR = REPO_ROOT / "things_similarity"
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
BEHAVIORAL_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"

# SimLex-999 (consolidated under data/ in the restructure)
SIMLEX_PATH = BASE_DATA_DIR / "SimLex-999" / "SimLex-999.txt"

# Output
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# Hyperparameters (reasonable defaults; tune freely)
#------------------------------------------------------------------------------

# Vocab / tokenization
MIN_COUNT = 5
MAX_VOCAB = None          # or int like 200_000

# Word2Vec
EMBEDDING_DIM = 200
WINDOW_SIZE = 5
NEG_SAMPLES = 10
SUBSAMPLE_T = 1e-5        # set None to disable subsampling

# Training schedule
EPOCHS = 2
BATCH_SIZE = 1024
LR = 2e-3
BATCHES_PER_EPOCH = 10_000   # caps training per epoch (streaming corpus)

# RSR (THINGS similarity alignment)
RSR_WEIGHT = 0.01
RSR_EVERY_N_BATCHES = 10
RSR_PAIRS_PER_STEP = 5_000
SOFT_RANK_STRENGTH = 2.0

# Optional: THINGS-focused W2V (leave 0.0 if you only want "add one THINGS loss" via RSR)
THINGS_W2V_WEIGHT = 0.0

# Corpus loading limits (for debugging)
MAX_JSON_FILES = None         # e.g. 2 for quick test
MAX_ARTICLES = None           # e.g. 5000 for quick test

print("[config]")
print(f"  WIKI_DIR={WIKI_DIR}")
print(f"  MIN_COUNT={MIN_COUNT} | EMBEDDING_DIM={EMBEDDING_DIM} | WINDOW={WINDOW_SIZE} | NEG={NEG_SAMPLES}")
print(f"  EPOCHS={EPOCHS} | BATCH_SIZE={BATCH_SIZE} | LR={LR} | BATCHES_PER_EPOCH={BATCHES_PER_EPOCH}")
print(f"  RSR_WEIGHT={RSR_WEIGHT} | RSR_EVERY={RSR_EVERY_N_BATCHES} | RSR_PAIRS={RSR_PAIRS_PER_STEP}")
print(f"  THINGS_W2V_WEIGHT={THINGS_W2V_WEIGHT}")

#------------------------------------------------------------------------------
# Tokenization + streaming Wikipedia sentence iterator
#------------------------------------------------------------------------------

_token_re = re.compile(r"[^a-zA-Z\s]+")

def simple_tokenize(text: str):
    text = text.lower()
    text = _token_re.sub(" ", text)
    return text.split()

def iter_wiki_sentences_json(
    wiki_dir: Path,
    max_json_files=None,
    max_articles=None,
):
    """
    Stream tokenized sentences from a directory of JSON files.
    Each JSON file contains an array of articles. Each article has a 'text' field.

    This is streaming to avoid holding the whole corpus in RAM.
    """
    json_files = sorted(wiki_dir.glob("*.json"))
    if max_json_files is not None:
        json_files = json_files[:max_json_files]

    article_seen = 0
    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            try:
                articles = json.load(f)
            except Exception as e:
                print(f"[warn] Failed to read {jf}: {e}")
                continue

        for art in articles:
            if max_articles is not None and article_seen >= max_articles:
                return
            article_seen += 1

            text = art.get("text", "")
            if not text:
                continue

            # cheap sentence split; good enough for W2V
            for sent in text.split(". "):
                toks = simple_tokenize(sent)
                if len(toks) >= 2:
                    yield toks

def sentence_stream_factory():
    # Factory that returns a *fresh* generator each time (important for multiple passes / epochs)
    return iter_wiki_sentences_json(
        WIKI_DIR,
        max_json_files=MAX_JSON_FILES,
        max_articles=MAX_ARTICLES,
    )

#------------------------------------------------------------------------------
# Build vocab (1st pass over the corpus)
#------------------------------------------------------------------------------

def build_vocab_from_stream(stream, min_count=5, max_vocab=None):
    counts = Counter()
    for toks in tqdm(stream, desc="Counting vocab (stream pass 1)"):
        counts.update(toks)

    # Filter + sort
    items = [(w, c) for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    if max_vocab is not None:
        items = items[:max_vocab]

    # Reserve 0 for <UNK> to make filtering easier
    vocab = ["<UNK>"] + [w for w, _ in items]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    # counts aligned to vocab indices
    idx_counts = np.zeros(len(vocab), dtype=np.int64)
    idx_counts[0] = 1  # arbitrary for <UNK>
    for w, c in items:
        idx_counts[word2idx[w]] = c

    return word2idx, idx2word, idx_counts

print("\n" + "=" * 70)
print("STEP 1: Build vocabulary (streaming over Wikipedia)")
print("=" * 70)

word2idx, idx2word, idx_counts = build_vocab_from_stream(
    sentence_stream_factory(),
    min_count=MIN_COUNT,
    max_vocab=MAX_VOCAB,
)

VOCAB_SIZE = len(word2idx)
print(f"[info] vocab_size={VOCAB_SIZE:,} (min_count={MIN_COUNT})")

#------------------------------------------------------------------------------
# Subsampling + negative sampling distributions
#------------------------------------------------------------------------------

def make_unigram_dist(idx_counts: np.ndarray, power: float = 0.75) -> torch.Tensor:
    """
    Negative sampling distribution ~ count^0.75
    Returns a normalized torch tensor on CPU (use torch.multinomial for sampling).
    """
    freqs = idx_counts.astype(np.float64)
    freqs[0] = 0.0  # don't sample <UNK> as negative
    p = np.power(freqs, power)
    p = p / (p.sum() + 1e-12)
    return torch.tensor(p, dtype=torch.float32)

NEG_DIST = make_unigram_dist(idx_counts)

def make_subsampling_keep_probs(idx_counts: np.ndarray, t: float = 1e-5) -> np.ndarray:
    """
    Mikolov subsampling: keep_prob = min(1, sqrt(t/f) + t/f)
    where f is word frequency.
    """
    freqs = idx_counts / idx_counts.sum()
    keep = np.ones_like(freqs, dtype=np.float64)
    # avoid div0 and ignore <UNK>
    mask = freqs > 0
    keep[mask] = np.minimum(1.0, (np.sqrt(t / freqs[mask]) + (t / freqs[mask])))
    keep[0] = 0.0
    return keep

KEEP_PROB = None
if SUBSAMPLE_T is not None:
    KEEP_PROB = make_subsampling_keep_probs(idx_counts, t=SUBSAMPLE_T)

def tokens_to_indices(tokens):
    """
    Map tokens -> indices, filter OOV to <UNK> (then we drop <UNK>).
    Apply optional subsampling.
    """
    idxs = []
    for w in tokens:
        i = word2idx.get(w, 0)
        if i == 0:
            continue
        if KEEP_PROB is not None:
            if random.random() > KEEP_PROB[i]:
                continue
        idxs.append(i)
    return idxs

#------------------------------------------------------------------------------
# Generate skip-gram pairs (streaming) + batch builder
#------------------------------------------------------------------------------

def iter_skipgram_pairs(sentence_stream, window_size=5):
    """
    Yields (target_idx, context_idx) pairs for skip-gram training.
    Streaming; does not store corpus.
    """
    for toks in sentence_stream:
        idxs = tokens_to_indices(toks)
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
    """
    Yield batches of (targets, pos_contexts) as torch tensors.
    """
    targets = []
    contexts = []
    for t, c in pair_iter:
        targets.append(t)
        contexts.append(c)
        if len(targets) >= batch_size:
            yield torch.tensor(targets, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)
            targets, contexts = [], []
    # drop remainder for simplicity (fine for streaming)

#------------------------------------------------------------------------------
# Model: Skip-gram with negative sampling (FROM SCRATCH)
#------------------------------------------------------------------------------

class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Init similar to word2vec-ish small uniform
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.uniform_(self.out_embed.weight, -bound, bound)

    def forward(self, target_idx, pos_ctx_idx, neg_ctx_idx):
        """
        target_idx: (B,)
        pos_ctx_idx: (B,)
        neg_ctx_idx: (B, K)
        """
        v = self.in_embed(target_idx)                 # (B, D)
        u_pos = self.out_embed(pos_ctx_idx)           # (B, D)
        u_neg = self.out_embed(neg_ctx_idx)           # (B, K, D)

        pos_logits = (v * u_pos).sum(dim=1)           # (B,)
        neg_logits = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (B, K)

        return pos_logits, neg_logits

def w2v_neg_sampling_loss(pos_logits, neg_logits):
    """
    Negative sampling objective:
      log sigma(pos) + sum_k log sigma(-neg_k)
    (we minimize negative of that)
    """
    pos_loss = F.logsigmoid(pos_logits).mean()
    neg_loss = F.logsigmoid(-neg_logits).mean()
    return -(pos_loss + neg_loss)

#------------------------------------------------------------------------------
# RSR: soft rank + soft Spearman (differentiable)
#------------------------------------------------------------------------------

def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """
    Differentiable approximation to ranks using pairwise sigmoid comparisons.
    x: (n,)
    returns: (n,)
    """
    x = x.flatten()
    diffs = x.unsqueeze(1) - x.unsqueeze(0)   # (n, n)
    soft_comparisons = torch.sigmoid(regularization_strength * diffs)
    ranks = soft_comparisons.sum(dim=1)
    return ranks

def soft_spearman(pred: torch.Tensor, target: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """
    Differentiable Spearman correlation via soft ranks.
    Returns correlation ~ [-1, 1]
    """
    pr = soft_rank(pred, regularization_strength)
    tr = soft_rank(target, regularization_strength)

    pr = pr - pr.mean()
    tr = tr - tr.mean()

    pr = pr / (pr.norm() + 1e-8)
    tr = tr / (tr.norm() + 1e-8)

    return (pr * tr).sum()

#------------------------------------------------------------------------------
# THINGS loader + RSR pair sampler
#------------------------------------------------------------------------------

def load_things_words(path: Path):
    """
    THINGS unique_id.txt typically includes one concept per line.
    """
    with path.open("r", encoding="utf-8") as f:
        words = [ln.strip() for ln in f if ln.strip()]
    # normalize to match tokenization
    words = [w.lower() for w in words]
    return words

def load_spose_similarity(path: Path):
    """
    Loads spose_similarity.mat and returns similarity matrix.
    Variable name is often 'spose_sim' or similar; we defensively pick the first 2D array.
    """
    mat = sio.loadmat(path)
    # Find first 2D numeric array
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == v.shape[1]:
            return v
    raise ValueError(f"Could not find square similarity matrix in {path}")

print("\n" + "=" * 70)
print("STEP 2: Load THINGS + behavioral similarity")
print("=" * 70)

things_words = load_things_words(THINGS_WORDS_PATH)
spose_sim = load_spose_similarity(BEHAVIORAL_SIM_PATH)

# Build mapping: THINGS concepts -> vocab indices (only those present in our vocab)
valid_concepts = []
valid_vocab_indices = []
valid_things_indices = []

things_word_to_i = {w: i for i, w in enumerate(things_words)}
for w in things_words:
    vi = word2idx.get(w, 0)
    if vi != 0:
        valid_concepts.append(w)
        valid_vocab_indices.append(vi)
        valid_things_indices.append(things_word_to_i[w])

valid_vocab_indices = np.array(valid_vocab_indices, dtype=np.int64)
valid_things_indices = np.array(valid_things_indices, dtype=np.int64)

print(f"[info] THINGS concepts total={len(things_words):,}")
print(f"[info] THINGS concepts in vocab={len(valid_concepts):,}")

# Extract the submatrix of behavioral similarity for the concepts we can train/eval on
THINGS_SIM_SUB = spose_sim[np.ix_(valid_things_indices, valid_things_indices)].astype(np.float32)

# Precompute upper-tri indices for uniform random pair sampling
# (exclude diagonal)
N_TH = THINGS_SIM_SUB.shape[0]
tri_u = np.triu_indices(N_TH, k=1)
ALL_PAIR_COUNT = len(tri_u[0])
print(f"[info] Available THINGS pairs (upper-tri)={ALL_PAIR_COUNT:,}")

def sample_things_pairs(num_pairs: int):
    """
    Returns:
      vocab_i: (P,)
      vocab_j: (P,)
      target_sim: (P,)
    """
    idx = np.random.randint(0, ALL_PAIR_COUNT, size=num_pairs)
    ai = tri_u[0][idx]
    aj = tri_u[1][idx]

    vocab_i = valid_vocab_indices[ai]
    vocab_j = valid_vocab_indices[aj]
    target_sim = THINGS_SIM_SUB[ai, aj]

    return (
        torch.tensor(vocab_i, dtype=torch.long, device=DEVICE),
        torch.tensor(vocab_j, dtype=torch.long, device=DEVICE),
        torch.tensor(target_sim, dtype=torch.float32, device=DEVICE),
    )

#------------------------------------------------------------------------------
# Training helpers
#------------------------------------------------------------------------------

@torch.no_grad()
def cosine_sim_from_in_embeddings(model: SkipGramWord2Vec, idx_a: torch.Tensor, idx_b: torch.Tensor):
    """
    cosine similarity between in-embeddings (common choice for W2V)
    idx_a/idx_b: (P,)
    returns: (P,)
    """
    va = model.in_embed(idx_a)
    vb = model.in_embed(idx_b)
    va = va / (va.norm(dim=1, keepdim=True) + 1e-8)
    vb = vb / (vb.norm(dim=1, keepdim=True) + 1e-8)
    return (va * vb).sum(dim=1)

def cosine_sim_from_in_embeddings_grad(model: SkipGramWord2Vec, idx_a: torch.Tensor, idx_b: torch.Tensor):
    """
    same as above, but with gradients enabled (for RSR loss).
    """
    va = model.in_embed(idx_a)
    vb = model.in_embed(idx_b)
    va = va / (va.norm(dim=1, keepdim=True) + 1e-8)
    vb = vb / (vb.norm(dim=1, keepdim=True) + 1e-8)
    return (va * vb).sum(dim=1)

def train_one_epoch_streaming(
    model: SkipGramWord2Vec,
    optimizer: optim.Optimizer,
    sentence_stream_fn,
    batches_per_epoch: int,
    batch_size: int,
    window_size: int,
    neg_samples: int,
    neg_dist: torch.Tensor,
    rsr_weight: float = 0.0,
    rsr_every_n: int = 10,
    rsr_pairs_per_step: int = 5000,
    soft_rank_strength: float = 2.0,
    things_w2v_weight: float = 0.0,
):
    model.train()

    # streaming pair iterator (fresh)
    pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size=window_size)
    batch_iter = batch_pairs(pair_iter, batch_size=batch_size)

    total_loss = 0.0
    total_w2v = 0.0
    total_rsr = 0.0
    total_thw2v = 0.0

    pbar = tqdm(range(batches_per_epoch), desc="training", leave=False)
    for b in pbar:
        try:
            tgt, ctx = next(batch_iter)
        except StopIteration:
            # corpus exhausted; restart stream mid-epoch (fine for huge corpora)
            pair_iter = iter_skipgram_pairs(sentence_stream_fn(), window_size=window_size)
            batch_iter = batch_pairs(pair_iter, batch_size=batch_size)
            tgt, ctx = next(batch_iter)

        tgt = tgt.to(DEVICE)
        ctx = ctx.to(DEVICE)

        # negatives: (B, K)
        neg = torch.multinomial(neg_dist, num_samples=tgt.shape[0] * neg_samples, replacement=True)
        neg = neg.view(tgt.shape[0], neg_samples).to(DEVICE)

        pos_logits, neg_logits = model(tgt, ctx, neg)
        loss_w2v = w2v_neg_sampling_loss(pos_logits, neg_logits)

        # Optional THINGS-focused W2V term (kept as a simple hook; default 0.0)
        # Here we just reuse the same w2v loss; in a more elaborate setup you’d bias sampling
        # toward THINGS words/sentences.
        loss_things_w2v = loss_w2v.detach() * 0.0
        if things_w2v_weight > 0.0:
            loss_things_w2v = loss_w2v  # placeholder: you can implement a THINGS-biased sampler later

        # RSR term (every N batches)
        loss_rsr = torch.tensor(0.0, device=DEVICE)
        if rsr_weight > 0.0 and (b % rsr_every_n == 0):
            i_idx, j_idx, target_sim = sample_things_pairs(rsr_pairs_per_step)
            pred_sim = cosine_sim_from_in_embeddings_grad(model, i_idx, j_idx)
            rho = soft_spearman(pred_sim, target_sim, regularization_strength=soft_rank_strength)
            loss_rsr = 1.0 - rho  # maximize rho -> minimize (1-rho)

        loss = loss_w2v + (things_w2v_weight * loss_things_w2v) + (rsr_weight * loss_rsr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_w2v += float(loss_w2v.item())
        total_rsr += float(loss_rsr.item()) if rsr_weight > 0.0 else 0.0
        total_thw2v += float(loss_things_w2v.item()) if things_w2v_weight > 0.0 else 0.0

        if (b + 1) % 200 == 0:
            pbar.set_postfix({
                "loss": total_loss / (b + 1),
                "w2v": total_w2v / (b + 1),
                "rsr": (total_rsr / max(1, (b // rsr_every_n) + 1)) if rsr_weight > 0.0 else 0.0
            })

    return {
        "loss": total_loss / batches_per_epoch,
        "w2v": total_w2v / batches_per_epoch,
        "rsr": total_rsr / max(1, (batches_per_epoch // rsr_every_n)) if rsr_weight > 0.0 else 0.0,
        "things_w2v": total_thw2v / batches_per_epoch if things_w2v_weight > 0.0 else 0.0,
    }

#------------------------------------------------------------------------------
# STEP 3: Create a single shared random initialization (FAIR comparison)
#------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: Create shared init state for fair Vanilla vs RSR comparison")
print("=" * 70)

init_model = SkipGramWord2Vec(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
init_state = deepcopy(init_model.state_dict())
del init_model
print("[info] Shared init_state captured.")

#------------------------------------------------------------------------------
# STEP 4: Train Vanilla model (Wikipedia only)
#------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4: Train Vanilla Word2Vec (Wikipedia only)")
print("=" * 70)

vanilla_model = SkipGramWord2Vec(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
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
        neg_dist=NEG_DIST,
        rsr_weight=0.0,                 # <-- vanilla: no RSR
        things_w2v_weight=0.0,
    )
    print(f"[vanilla] epoch {ep}/{EPOCHS} | loss={stats['loss']:.4f} | w2v={stats['w2v']:.4f}")

vanilla_path = MODELS_DIR / "vanilla_w2v.pt"
torch.save(
    {
        "state_dict": vanilla_model.state_dict(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "embedding_dim": EMBEDDING_DIM,
        "config": {
            "MIN_COUNT": MIN_COUNT,
            "WINDOW_SIZE": WINDOW_SIZE,
            "NEG_SAMPLES": NEG_SAMPLES,
            "SUBSAMPLE_T": SUBSAMPLE_T,
        },
    },
    vanilla_path,
)
print(f"[saved] {vanilla_path}")

#------------------------------------------------------------------------------
# STEP 5: Train RSR model (Wikipedia + THINGS RSR loss)
#------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5: Train RSR Word2Vec (Wikipedia + THINGS-informed loss)")
print("=" * 70)

rsr_model = SkipGramWord2Vec(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
rsr_model.load_state_dict(deepcopy(init_state))  # <-- same init as vanilla

rsr_opt = optim.Adam(rsr_model.parameters(), lr=LR)

for ep in range(1, EPOCHS + 1):
    stats = train_one_epoch_streaming(
        model=rsr_model,
        optimizer=rsr_opt,
        sentence_stream_fn=sentence_stream_factory,
        batches_per_epoch=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        neg_samples=NEG_SAMPLES,
        neg_dist=NEG_DIST,
        rsr_weight=RSR_WEIGHT,
        rsr_every_n=RSR_EVERY_N_BATCHES,
        rsr_pairs_per_step=RSR_PAIRS_PER_STEP,
        soft_rank_strength=SOFT_RANK_STRENGTH,
        things_w2v_weight=THINGS_W2V_WEIGHT,
    )
    print(
        f"[rsr] epoch {ep}/{EPOCHS} | loss={stats['loss']:.4f} | w2v={stats['w2v']:.4f} | "
        f"rsr={stats['rsr']:.4f}"
    )

rsr_path = MODELS_DIR / "rsr_w2v.pt"
torch.save(
    {
        "state_dict": rsr_model.state_dict(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "embedding_dim": EMBEDDING_DIM,
        "config": {
            "MIN_COUNT": MIN_COUNT,
            "WINDOW_SIZE": WINDOW_SIZE,
            "NEG_SAMPLES": NEG_SAMPLES,
            "SUBSAMPLE_T": SUBSAMPLE_T,
            "RSR_WEIGHT": RSR_WEIGHT,
            "RSR_EVERY_N_BATCHES": RSR_EVERY_N_BATCHES,
            "RSR_PAIRS_PER_STEP": RSR_PAIRS_PER_STEP,
            "SOFT_RANK_STRENGTH": SOFT_RANK_STRENGTH,
            "THINGS_W2V_WEIGHT": THINGS_W2V_WEIGHT,
        },
    },
    rsr_path,
)
print(f"[saved] {rsr_path}")

#------------------------------------------------------------------------------
# STEP 6: SimLex-999 evaluation (Spearman correlation)
#------------------------------------------------------------------------------

def load_simlex(path: Path):
    """
    Supports common SimLex formats:
      - TSV with header: word1 word2 SimLex999
      - CSV with 'word1','word2','SimLex999' columns
    """
    # Try pandas auto-detect
    df = pd.read_csv(path, sep=None, engine="python")
    cols = [c.lower() for c in df.columns]

    # Heuristic column picks
    if "word1" in cols and "word2" in cols:
        w1_col = df.columns[cols.index("word1")]
        w2_col = df.columns[cols.index("word2")]
    else:
        # fallback: first two columns
        w1_col, w2_col = df.columns[:2]

    # score column
    score_col = None
    for candidate in ["simlex999", "simlex", "score", "similarity"]:
        if candidate in cols:
            score_col = df.columns[cols.index(candidate)]
            break
    if score_col is None:
        score_col = df.columns[2]  # common layout

    df = df[[w1_col, w2_col, score_col]].copy()
    df.columns = ["word1", "word2", "score"]
    df["word1"] = df["word1"].astype(str).str.lower()
    df["word2"] = df["word2"].astype(str).str.lower()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    return df

@torch.no_grad()
def simlex_spearman(model: SkipGramWord2Vec, simlex_df: pd.DataFrame, restrict_to_things: bool = False):
    """
    Compute Spearman between model cosine similarities and SimLex human scores.
    If restrict_to_things=True, only include pairs where both words are THINGS concepts in vocab.
    """
    model.eval()

    # Get in-embedding matrix
    W = model.in_embed.weight.detach()  # (V, D)

    # Normalize for cosine
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)

    things_set = set(valid_concepts)

    sims = []
    scores = []
    covered = 0

    for _, row in simlex_df.iterrows():
        w1 = row["word1"]
        w2 = row["word2"]
        if restrict_to_things and (w1 not in things_set or w2 not in things_set):
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
        return {
            "rho": np.nan,
            "p": np.nan,
            "n": covered,
        }

    rho, p = spearmanr(sims, scores)
    return {
        "rho": float(rho),
        "p": float(p),
        "n": int(covered),
    }

print("\n" + "=" * 70)
print("STEP 6: SimLex-999 evaluation")
print("=" * 70)

if not SIMLEX_PATH.exists():
    print(f"[warn] SimLex file not found at: {SIMLEX_PATH}")
    print("       Update SIMLEX_PATH at the top of the notebook/script.")
else:
    simlex_df = load_simlex(SIMLEX_PATH)
    print(f"[info] Loaded SimLex rows: {len(simlex_df):,}")

    v_all = simlex_spearman(vanilla_model, simlex_df, restrict_to_things=False)
    r_all = simlex_spearman(rsr_model, simlex_df, restrict_to_things=False)

    v_th = simlex_spearman(vanilla_model, simlex_df, restrict_to_things=True)
    r_th = simlex_spearman(rsr_model, simlex_df, restrict_to_things=True)

    print("\n[SimLex-999 | all covered pairs]")
    print(f"  vanilla: rho={v_all['rho']:.4f} (n={v_all['n']})")
    print(f"  rsr:     rho={r_all['rho']:.4f} (n={r_all['n']})")

    print("\n[SimLex-999 | THINGS-only pairs]")
    print(f"  vanilla: rho={v_th['rho']:.4f} (n={v_th['n']})")
    print(f"  rsr:     rho={r_th['rho']:.4f} (n={r_th['n']})")

#------------------------------------------------------------------------------
# STEP 7 (optional): quick qualitative nearest neighbors
#------------------------------------------------------------------------------

@torch.no_grad()
def nearest_neighbors(model: SkipGramWord2Vec, query_word: str, topk: int = 10):
    model.eval()
    query_word = query_word.lower()
    qi = word2idx.get(query_word, 0)
    if qi == 0:
        return None

    W = model.in_embed.weight.detach()
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    qv = Wn[qi:qi+1]  # (1, D)

    sims = torch.matmul(Wn, qv.t()).squeeze(1)  # (V,)
    # exclude itself and <UNK>
    sims[qi] = -1e9
    sims[0] = -1e9

    vals, idxs = torch.topk(sims, k=topk)
    return [(idx2word[int(i)], float(v)) for i, v in zip(idxs.cpu(), vals.cpu())]

test_words = ["cat", "dog", "king", "queen", "car", "bicycle"]
for w in test_words:
    nn_v = nearest_neighbors(vanilla_model, w, topk=8)
    nn_r = nearest_neighbors(rsr_model, w, topk=8)
    if nn_v is None or nn_r is None:
        print(f"\n[w={w}] not in vocab")
        continue
    print(f"\n[w={w}] vanilla:", nn_v)
    print(f"[w={w}] rsr:    ", nn_r)
