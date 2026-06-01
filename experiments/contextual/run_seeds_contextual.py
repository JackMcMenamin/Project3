"""
RSR Multi-Seed Experiment — Contextual-Vector Edition

The same 10-seed RSR experiment as `run_bert_seeds.py` / `run_gpt2_seeds.py`,
but every word's representation is the **cached contextual vector** produced
by `extract_contextual_vectors.py` (mean-pooled over up to 50 Wikipedia
sentences containing the word). The transformer backbones are not loaded at
all here — only a 768→128 projection head is trainable, and that head is the
sole RSR-optimised parameter set.

Compared to the original scripts:
  * BERT/GPT-2 are NOT loaded.
  * `get_word_embedding` is replaced with a cache lookup.
  * SimLex pairs whose words are not in the cache are dropped from
    evaluation and reported separately (no fallback encoding).
  * Everything else — 10 seeds, 200 epochs, lr=1e-3, sample size 10000,
    soft-Spearman loss, the both/one/neither-in-RSR partition analysis,
    Excel output — is preserved so the new numbers are directly comparable
    with the existing Table 2 in the paper.

Usage:
    python run_seeds_contextual.py --model bert
    python run_seeds_contextual.py --model gpt2
"""
from __future__ import annotations

import argparse
import gc
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

try:
    import torchsort
    HAS_TORCHSORT = True
except ImportError:
    HAS_TORCHSORT = False


# ==============================================================================
# Configuration
# ==============================================================================

SEEDS = list(range(1, 11))

ROOT = Path(__file__).resolve().parents[2]  # experiments/contextual/ -> repo root
BASE_DATA_DIR = ROOT / "data"
THINGS_DIR = ROOT / "things_similarity"
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"
MEN_PATH = BASE_DATA_DIR / "MEN" / "MEN" / "MEN_dataset_lemma_form_full"
SIMVERB_PATH = BASE_DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
SIMLEX_PATH = BASE_DATA_DIR / "SimLex-999" / "SimLex-999.txt"

# Hyperparameters — kept identical to the existing scripts so the comparison
# is honest. Note: BATCH_SIZE is unused now since cache lookups are O(1).
RSR_EPOCHS = 200
RSR_LR = 1e-3
RSR_SAMPLE_SIZE = 10000
SOFT_RANK_STRENGTH = 1.0
PROJECTION_DIM = 128
INPUT_DIM = 768  # both BERT-base and GPT-2 are 768-d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Loss helpers (verbatim from run_bert_seeds.py)
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_rank_custom(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    soft_comparisons = torch.sigmoid(diff * regularization_strength)
    ranks = soft_comparisons.sum(dim=2) + 0.5
    return ranks.squeeze(0)


def soft_spearman(pred: torch.Tensor, target: torch.Tensor,
                  regularization_strength: float = SOFT_RANK_STRENGTH) -> torch.Tensor:
    if HAS_TORCHSORT:
        pred_rank = torchsort.soft_rank(
            pred.unsqueeze(0), regularization_strength=regularization_strength
        ).squeeze(0)
        target_rank = torchsort.soft_rank(
            target.unsqueeze(0), regularization_strength=regularization_strength
        ).squeeze(0)
    else:
        pred_rank = soft_rank_custom(pred, regularization_strength)
        target_rank = soft_rank_custom(target, regularization_strength)
    pred_centered = pred_rank - pred_rank.mean()
    target_centered = target_rank - target_rank.mean()
    cov = (pred_centered * target_centered).mean()
    return cov / (pred_centered.std() * target_centered.std() + 1e-8)


# ==============================================================================
# Model: cached vectors + trainable projection head
# ==============================================================================

class ContextualEmbeddings(nn.Module):
    """Frozen cached vectors + trainable projection head.

    The cached 768-d contextual vectors are stored in a non-trainable buffer
    (one row per known target). The trainable parameters are limited to the
    768→128 projection head, which is what RSR optimises.
    """
    def __init__(self, vectors: dict[str, np.ndarray], projection_dim: int = PROJECTION_DIM):
        super().__init__()
        # Build deterministic key→row mapping.
        self.keys: list[str] = sorted(vectors.keys())
        self.key_to_idx: dict[str, int] = {k: i for i, k in enumerate(self.keys)}

        matrix = np.stack([vectors[k] for k in self.keys]).astype(np.float32)
        # Frozen non-trainable buffer of contextual vectors.
        self.register_buffer("table", torch.from_numpy(matrix))

        self.projection = nn.Linear(INPUT_DIM, projection_dim)
        self.to(DEVICE)

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_idx

    def lookup_indices(self, words: list[str]) -> torch.Tensor:
        return torch.tensor(
            [self.key_to_idx[w] for w in words], dtype=torch.long, device=DEVICE,
        )

    def embed(self, words: list[str]) -> torch.Tensor:
        """Project the cached vectors for `words`. Caller must filter to in-cache words."""
        idx = self.lookup_indices(words)
        raw = self.table[idx]                          # (N, 768)
        return self.projection(raw)                    # (N, projection_dim)

    def embed_dict(self, words: list[str]) -> dict[str, torch.Tensor]:
        present = [w for w in words if w in self.key_to_idx]
        if not present:
            return {}
        out = self.embed(present)
        return {w: out[i] for i, w in enumerate(present)}


# ==============================================================================
# Data loading — mirrors the existing scripts so RSR vocabulary stays identical
# ==============================================================================

def load_men_pairs():
    pairs = []
    if not MEN_PATH.exists():
        return pairs
    with open(MEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                w1, w2 = parts[0].split("-")[0], parts[1].split("-")[0]
                pairs.append((w1, w2, float(parts[2])))
    return pairs


def load_simverb_pairs():
    pairs = []
    if not SIMVERB_PATH.exists():
        return pairs
    with open(SIMVERB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                pairs.append((parts[0], parts[1], float(parts[3])))
    return pairs


def load_things_pairs():
    """Load THINGS pairs.

    The original scripts replaced spaces with underscores in concept names so
    they could be passed straight to the tokenizer. We instead emit the
    space-separated form to match the cache keys, and drop sense-suffixed
    concepts (baton1, bow2, ...) and any concept missing from the cache.
    """
    words = []
    with open(THINGS_WORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().replace("_", " ")  # match cache convention
            words.append(w)
    mat = sio.loadmat(str(THINGS_SIM_PATH))
    sim = mat["spose_sim"]
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            pairs.append((words[i], words[j], float(sim[i, j])))
    return pairs


def load_all_rsr_datasets():
    men = load_men_pairs()
    sv = load_simverb_pairs()
    th = load_things_pairs()

    def normalize(pairs):
        if not pairs:
            return pairs
        scores = [p[2] for p in pairs]
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-8:
            return pairs
        return [(p[0], p[1], (p[2] - lo) / (hi - lo)) for p in pairs]

    men, sv, th = normalize(men), normalize(sv), normalize(th)
    all_pairs = men + sv + th
    rsr_words = {w for triple in all_pairs for w in triple[:2]}
    info = {
        "men": len(men), "simverb": len(sv), "things": len(th),
        "total": len(all_pairs), "unique_words": len(rsr_words),
    }
    return all_pairs, info, rsr_words


def load_simlex(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            pairs.append((parts[0], parts[1], float(parts[3])))
    return pairs


def load_cached_vectors(model_name: str) -> dict[str, np.ndarray]:
    cache_path = ROOT / "artifacts" / "cache" / model_name / "vectors.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache found at {cache_path}. Run extract_contextual_vectors.py "
            f"--model {model_name} first."
        )
    data = np.load(cache_path)
    # Vectors are stored under slug keys (e.g. 'air_conditioner'); the cache
    # uses underscores to be filesystem-safe. The RSR pipeline uses spaces for
    # MWEs (matching the manifest target column). Convert slug→space form so
    # callers can index with whatever case the dataset gives them.
    out: dict[str, np.ndarray] = {}
    for slug in data.files:
        out[slug] = data[slug]
        # Also expose the space form for MWEs.
        if "_" in slug:
            spaced = slug.replace("_", " ")
            out[spaced] = data[slug]
    return out


# ==============================================================================
# Training and evaluation
# ==============================================================================

def train_rsr(model: ContextualEmbeddings, all_pairs, n_epochs: int,
              sample_size: int, lr: float) -> None:
    # Filter to pairs both of whose words are cached.
    valid = [(w1, w2, s) for w1, w2, s in all_pairs if w1 in model and w2 in model]
    n_dropped = len(all_pairs) - len(valid)
    print(f"  RSR pairs after cache filter: {len(valid)} "
          f"(dropped {n_dropped} for missing vectors)")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        sample = random.sample(valid, sample_size) if len(valid) > sample_size else valid

        unique_words = list({w for triple in sample for w in triple[:2]})
        embs = model.embed_dict(unique_words)

        model_sims, human_sims = [], []
        for w1, w2, s in sample:
            cos = torch.nn.functional.cosine_similarity(
                embs[w1].unsqueeze(0), embs[w2].unsqueeze(0)
            )
            model_sims.append(cos)
            human_sims.append(s)

        if len(model_sims) < 10:
            continue

        ms = torch.cat(model_sims)
        hs = torch.tensor(human_sims, device=DEVICE)
        rho = soft_spearman(ms, hs)
        loss = 1 - rho

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, rho={rho.item():.4f}")


def evaluate_simlex(model: ContextualEmbeddings, simlex_pairs, rsr_words):
    """Evaluate on SimLex-999, partitioned by RSR-vocabulary overlap.

    Pairs whose words are not in the cached vector store are dropped from
    every category and counted under `n_missing` for transparency.
    """
    model.eval()
    categories = defaultdict(list)
    for w1, w2, s in simlex_pairs:
        cat = ("both_in_rsr" if (w1 in rsr_words and w2 in rsr_words)
               else "one_in_rsr" if (w1 in rsr_words or w2 in rsr_words)
               else "neither_in_rsr")
        categories["all"].append((w1, w2, s))
        categories[cat].append((w1, w2, s))

    results = {}
    for cat, pairs in categories.items():
        words = list({w for triple in pairs for w in triple[:2]})
        present = [w for w in words if w in model]
        n_missing_words = len(words) - len(present)

        with torch.no_grad():
            embs = model.embed_dict(present)

        ms, hs, n_skipped = [], [], 0
        for w1, w2, s in pairs:
            if w1 not in embs or w2 not in embs:
                n_skipped += 1
                continue
            cos = torch.nn.functional.cosine_similarity(
                embs[w1].unsqueeze(0), embs[w2].unsqueeze(0)
            ).item()
            ms.append(cos)
            hs.append(s)

        if len(ms) < 2:
            results[cat] = {"n": 0, "rho": float("nan"), "n_skipped": n_skipped,
                            "n_missing_words": n_missing_words}
            continue
        rho, _ = spearmanr(hs, ms)
        results[cat] = {"n": len(ms), "rho": rho, "n_skipped": n_skipped,
                        "n_missing_words": n_missing_words}
    return results


# ==============================================================================
# Per-seed and main
# ==============================================================================

def run_single_seed(seed: int, vectors, all_pairs, rsr_words, simlex_pairs):
    print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
    set_seed(seed)
    model = ContextualEmbeddings(vectors)

    print("  Evaluating vanilla (random projection)...")
    vanilla = evaluate_simlex(model, simlex_pairs, rsr_words)

    print("  Training RSR...")
    train_rsr(model, all_pairs, RSR_EPOCHS, RSR_SAMPLE_SIZE, RSR_LR)

    print("  Evaluating RSR-tuned...")
    rsr = evaluate_simlex(model, simlex_pairs, rsr_words)

    res = {
        "seed": seed,
        "vanilla_all": vanilla["all"]["rho"],
        "vanilla_both": vanilla["both_in_rsr"]["rho"],
        "vanilla_one": vanilla["one_in_rsr"]["rho"],
        "vanilla_neither": vanilla["neither_in_rsr"]["rho"],
        "rsr_all": rsr["all"]["rho"],
        "rsr_both": rsr["both_in_rsr"]["rho"],
        "rsr_one": rsr["one_in_rsr"]["rho"],
        "rsr_neither": rsr["neither_in_rsr"]["rho"],
        "n_simlex_evaluated": rsr["all"]["n"],
        "n_simlex_skipped": rsr["all"]["n_skipped"],
    }
    for k in ["all", "both", "one", "neither"]:
        res[f"delta_{k}"] = res[f"rsr_{k}"] - res[f"vanilla_{k}"]

    print(f"\n  Seed {seed}: van={res['vanilla_all']:.4f} -> "
          f"rsr={res['rsr_all']:.4f}  (delta {res['delta_all']:+.4f})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "gpt2"], required=True)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Contextual RSR Multi-Seed Experiment — {args.model.upper()}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print(f"Torchsort: {HAS_TORCHSORT}")
    print("=" * 70)

    print(f"\nLoading cached vectors from artifacts/cache/{args.model}/vectors.npz ...")
    vectors = load_cached_vectors(args.model)
    print(f"  {len(vectors)} keys in cache (incl. spaced MWE aliases)")

    print("\nLoading datasets...")
    all_pairs, info, rsr_words = load_all_rsr_datasets()
    simlex = load_simlex(SIMLEX_PATH)
    print(f"  RSR pairs: {info['total']} (vocab {info['unique_words']})")
    print(f"  SimLex-999 pairs: {len(simlex)}")

    # Diagnostic: how much of each set lives in the cache?
    rsr_in = sum(1 for w in rsr_words if w in vectors)
    simlex_words = {w for p in simlex for w in p[:2]}
    sl_in = sum(1 for w in simlex_words if w in vectors)
    print(f"  RSR vocabulary in cache: {rsr_in}/{len(rsr_words)} "
          f"({100*rsr_in/len(rsr_words):.1f}%)")
    print(f"  SimLex vocabulary in cache: {sl_in}/{len(simlex_words)} "
          f"({100*sl_in/len(simlex_words):.1f}%)")

    results = []
    for seed in SEEDS:
        results.append(run_single_seed(seed, vectors, all_pairs, rsr_words, simlex))

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "results" / f"{args.model}_contextual_seeds_{timestamp}.xlsx"
    out_path.parent.mkdir(exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Summary mirroring the original scripts.
    print("\n" + "=" * 70)
    print(f"{args.model.upper()} CONTEXTUAL RSR -- SUMMARY")
    print("=" * 70)
    print(f"\n{'Seed':<6} {'Van All':>10} {'RSR All':>10} {'d All':>10} {'d Neither':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['seed']:<6} {r['vanilla_all']:>10.4f} {r['rsr_all']:>10.4f} "
              f"{r['delta_all']:>+10.4f} {r['delta_neither']:>+12.4f}")

    print("\nAggregated (Mean +/- Std):")
    for name, col in [("All - vanilla", "vanilla_all"),
                       ("All - RSR", "rsr_all"),
                       ("All - delta", "delta_all"),
                       ("Both in RSR - delta", "delta_both"),
                       ("One in RSR - delta", "delta_one"),
                       ("Neither in RSR - delta", "delta_neither")]:
        v = df[col].dropna()
        print(f"  {name:<24} {v.mean():>+8.4f}  +/- {v.std():.4f}")


if __name__ == "__main__":
    main()
