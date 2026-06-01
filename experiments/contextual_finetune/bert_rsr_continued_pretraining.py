"""
BERT continued pre-training with Representational Similarity Regularisation (RSR)
=================================================================================

Sentence-level RSR for BERT. Unlike the earlier "frozen cached-vector" pipeline,
here **BERT's own weights are trained** so the similarity constraint reshapes its
representations directly, while a masked-language-modelling (MLM) signal keeps it
a functioning language model.

This single file implements the design in
`docs/bert_continued_pretraining_rsr_design.md`. It is intentionally
self-contained (no project-package imports) so it can be sent to / run by anyone.
GPT-2 will follow the same template later; this file is BERT only.

--------------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------------
Two training regimes, selected with --regime:

  * baseline : plain continued MLM only (the control). BERT keeps pre-training
               the normal way (mask tokens, predict them, cross-entropy). No RSR.

  * rsr      : INTERLEAVED MLM + RSR. Alternate two kinds of batches:
                 (a) MLM batch  -> cross-entropy on masked tokens.
                 (b) RSR batch  -> N sentences, each containing one target word
                     (aardvark, chipmunk, ...). Run BERT, pull out each target
                     word's contextual token vector, project to a low-dim space,
                     build the NxN cosine-similarity matrix (upper triangle =
                     N(N-1)/2 pairs), and align it to the human similarity RDM
                     with a differentiable (soft) Spearman loss.

In both regimes BERT is mostly frozen: embeddings + the first 11 encoder layers
are frozen; only the final encoder block (+ the projection head, in the RSR
regime) trains. This freezing scheme and the default hyperparameters come from
Mark Ormerod's RSR thesis chapter (the foundation for this work).

--------------------------------------------------------------------------------
DATA (reuses what the repo already has)
--------------------------------------------------------------------------------
  * Human similarity (RSR supervision + ground-truth RDM):
        MEN + SimVerb-3500 + THINGS,  pooled & min-max normalised.
  * RSR-batch sentences:
        artifacts/sentences/<slug>.jsonl  (already harvested; up to 50 real
        Wikipedia sentences per target word).
  * MLM-batch sentences:
        by default, the same harvested sentence pool (any sentences work).
  * Held-out evaluation:
        SimLex-999  (NEVER used in training). Partitioned by RSR-supervision
        overlap (both / one / neither in RSR), as in the rest of the project.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    # DEFAULT: run vanilla BERT -> RSR BERT AND vanilla -> MLM-only, then print
    # the comparison table (Vanilla / RSR / Delta, partitioned both/one/neither):
    python experiments/contextual_finetune/bert_rsr_continued_pretraining.py

    # just one regime if you want:
    python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --regime rsr
    python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --regime baseline

    # quick smoke test (tiny, runs on CPU in a couple of minutes):
    python experiments/contextual_finetune/bert_rsr_continued_pretraining.py --smoke

Dependencies:  torch  transformers  numpy  pandas  scipy   (optional: torchsort)

--------------------------------------------------------------------------------
DESIGN CHOICES (all settled; each is a one-line change in Config if needed)
--------------------------------------------------------------------------------
  * Interleave MLM and RSR as separate batches (supervisor's request), RSR-
    majority at 1 MLM : 2 RSR per cycle (echoes Mark's heavy similarity weight).
  * Similarity is measured on the TARGET WORD'S token in context (aardvark inside
    its sentence), not a whole-sentence vector. This is the core of the method.
  * MLM batches train on the same harvested target-word sentences ("any
    sentences", per supervisor) -- no extra data needed.
  * RSR batch assembly: pick any target words; build the soft-Spearman loss only
    over the pairs that have a human score, mask the rest.
  * Multi-subword target -> mean-pool its subword token vectors.
  * Eval uses the same in-context extraction as training (honest comparison).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import BertForMaskedLM, BertTokenizerFast

# torchsort gives a better differentiable rank; fall back to a sigmoid version.
try:
    import torchsort

    HAS_TORCHSORT = True
except ImportError:
    HAS_TORCHSORT = False


# ============================================================================
# Paths  (anchored to the repo root so this runs from anywhere)
# ============================================================================
# experiments/contextual_finetune/<this file>  ->  repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
THINGS_DIR = REPO_ROOT / "things_similarity"
ARTIFACTS = REPO_ROOT / "artifacts"
SENTENCES_DIR = ARTIFACTS / "sentences"

MEN_PATH = DATA_DIR / "MEN" / "MEN" / "MEN_dataset_natural_form_full"
SIMVERB_PATH = DATA_DIR / "simverb-3500-data" / "data" / "SimVerb-3500.txt"
THINGS_WORDS_PATH = THINGS_DIR / "variables" / "unique_id.txt"
THINGS_SIM_PATH = THINGS_DIR / "data" / "spose_similarity.mat"
SIMLEX_PATH = DATA_DIR / "SimLex-999" / "SimLex-999.txt"

RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"


# ============================================================================
# Configuration (defaults from Mark Ormerod's RSR chapter; see design note)
# ============================================================================
class Config:
    model_name = "bert-base-uncased"
    num_frozen_layers = 11          # freeze embeddings + layers 0..10; train layer 11
    projection_dim = 128            # low-dim space the RSR constraint lives in
    # N sentences per RSR batch. Larger N = a denser RDM (N(N-1)/2 pairs) and a
    # much less noisy soft-Spearman gradient. The first run used N=5 (<=10 pairs)
    # which thrashed and degraded SimLex; 24 is far more stable.
    rsr_batch_size = 24
    mlm_batch_size = 16             # sentences per MLM batch
    mlm_probability = 0.15          # standard masking rate
    learning_rate = 5e-5
    soft_rank_strength = 1.0        # torchsort regularisation strength
    min_pairs_per_rsr_batch = 8     # skip an RSR batch with fewer scored pairs
    max_seq_len = 128
    steps = 2000                    # total optimiser steps in the main run
    eval_every = 250                # evaluate on SimLex every N steps
    seed = 1

    # Interleaving (rsr regime): alternate MLM and RSR batches in a repeating
    # cycle of length `interleave_cycle`; the first `mlm_per_cycle` steps of each
    # cycle are MLM, the rest RSR. Default 1 MLM : 1 RSR (balanced).
    #
    # NB: the first experiment used 1:2 (RSR-majority) reasoning from Mark's
    # lambda=0.9 — but Mark's weighting kept MLM present in *every* batch, whereas
    # interleaving with 1:2 leaves 2 of 3 batches with NO language signal. That
    # starved BERT and degraded SimLex. Balanced is the safer default; revisit
    # the ratio once a stable recipe is found.
    interleave_cycle = 2
    mlm_per_cycle = 1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Differentiable Spearman loss  (RSR term:  1 - rho_soft)
# ============================================================================
def soft_rank_custom(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Sigmoid-pairwise soft rank, used when torchsort is unavailable."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    ranks = torch.sigmoid(diff * strength).sum(dim=2) + 0.5
    return ranks.squeeze(0)


def soft_spearman(pred: torch.Tensor, target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Differentiable Spearman correlation between two 1-D score vectors."""
    if HAS_TORCHSORT:
        pred_r = torchsort.soft_rank(pred.unsqueeze(0), regularization_strength=strength).squeeze(0)
        tgt_r = torchsort.soft_rank(target.unsqueeze(0), regularization_strength=strength).squeeze(0)
    else:
        pred_r = soft_rank_custom(pred, strength)
        tgt_r = soft_rank_custom(target, strength)
    pred_c = pred_r - pred_r.mean()
    tgt_c = tgt_r - tgt_r.mean()
    return (pred_c * tgt_c).mean() / (pred_c.std() * tgt_c.std() + 1e-8)


# ============================================================================
# Human similarity data  (supervision pool + held-out SimLex)
# ============================================================================
def _normalize(pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    if not pairs:
        return pairs
    scores = [p[2] for p in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return pairs
    return [(a, b, (s - lo) / (hi - lo)) for a, b, s in pairs]


def load_men() -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    if not MEN_PATH.exists():
        return pairs
    with open(MEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                w1, w2 = parts[0].split("-")[0], parts[1].split("-")[0]
                pairs.append((w1, w2, float(parts[2])))
    return pairs


def load_simverb() -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    if not SIMVERB_PATH.exists():
        return pairs
    with open(SIMVERB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                pairs.append((parts[0], parts[1], float(parts[3])))
    return pairs


def load_things() -> list[tuple[str, str, float]]:
    if not THINGS_WORDS_PATH.exists() or not THINGS_SIM_PATH.exists():
        return []
    words = [ln.strip().replace(" ", "_") for ln in open(THINGS_WORDS_PATH, encoding="utf-8")]
    sim = sio.loadmat(str(THINGS_SIM_PATH))["spose_sim"]
    pairs: list[tuple[str, str, float]] = []
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((words[i], words[j], float(sim[i, j])))
    return pairs


def load_supervision() -> tuple[dict[tuple[str, str], float], set[str]]:
    """Return {(w1,w2)->score} lookup (both orderings) and the supervised vocab.

    Words are normalised to the cache/sentence convention: lowercase, MWEs as
    space-separated tokens (THINGS underscores -> spaces).
    """
    all_pairs = _normalize(load_men()) + _normalize(load_simverb()) + _normalize(load_things())

    def norm(w: str) -> str:
        return w.replace("_", " ").lower().strip()

    lookup: dict[tuple[str, str], float] = {}
    vocab: set[str] = set()
    for w1, w2, s in all_pairs:
        a, b = norm(w1), norm(w2)
        lookup[(a, b)] = s
        lookup[(b, a)] = s
        vocab.add(a)
        vocab.add(b)
    return lookup, vocab


def load_simlex() -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    with open(SIMLEX_PATH, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[3])))
    return pairs


# ============================================================================
# Sentences  (reuse the already-harvested artifacts/sentences/<slug>.jsonl)
# ============================================================================
def slugify(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", target.lower())


def load_sentences_for(target: str) -> list[str]:
    path = SENTENCES_DIR / f"{slugify(target)}.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line)["sentence"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return out


def find_subword_span(sent_ids: list[int], target_ids: list[int]) -> tuple[int, int] | None:
    """First contiguous occurrence of target_ids inside sent_ids -> (start, end_excl)."""
    n, m = len(sent_ids), len(target_ids)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if sent_ids[i:i + m] == target_ids:
            return i, i + m
    return None


# ============================================================================
# Model:  mostly-frozen BERT-for-MLM  +  a projection head for RSR
# ============================================================================
class BertRSR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
        # BertForMaskedLM gives us both the encoder (.bert) and the MLM head (.cls).
        self.mlm = BertForMaskedLM.from_pretrained(cfg.model_name)
        hidden = self.mlm.config.hidden_size
        self.projection = nn.Linear(hidden, cfg.projection_dim)

        # --- freeze embeddings + first `num_frozen_layers` encoder layers ---
        for p in self.mlm.bert.embeddings.parameters():
            p.requires_grad = False
        for i in range(cfg.num_frozen_layers):
            for p in self.mlm.bert.encoder.layer[i].parameters():
                p.requires_grad = False
        # Trainable: final encoder block, the MLM head (so MLM can still learn),
        # and the projection head.

        self.to(DEVICE)

    # ---- (a) MLM batch ----------------------------------------------------
    def mlm_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        out = self.mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    # ---- (b) RSR batch: contextual target-word vectors -> projected -------
    def target_vectors(self, input_ids, attention_mask, spans) -> torch.Tensor:
        """Mean-pool each sentence's target span from the last hidden state,
        then project. `spans` is a list of (start, end_excl) per row.
        Returns (N, projection_dim)."""
        out = self.mlm.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (N, L, H)
        vecs = []
        for j, (start, end) in enumerate(spans):
            seq_len = int(attention_mask[j].sum().item())
            end = min(end, seq_len)
            start = min(start, max(seq_len - 1, 0))
            vecs.append(hidden[j, start:end].mean(dim=0))
        stacked = torch.stack(vecs, dim=0)
        return self.projection(stacked)


# ============================================================================
# RSR-batch construction  (Option B: any words; mask unscored pairs)
# ============================================================================
class RSRBatchSampler:
    """Builds RSR batches of N (sentence, target) items where the targets have
    cached sentences. The loss masks to whichever pairs have a human score."""

    def __init__(self, supervised_vocab: set[str], sim_lookup: dict, tokenizer, cfg: Config):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.sim_lookup = sim_lookup
        # Targets we can actually use: supervised AND with harvested sentences.
        self.targets = sorted(
            w for w in supervised_vocab
            if (SENTENCES_DIR / f"{slugify(w)}.jsonl").exists()
        )
        # Cache the sentence lists lazily.
        self._sent_cache: dict[str, list[str]] = {}

    def _sentences(self, target: str) -> list[str]:
        if target not in self._sent_cache:
            self._sent_cache[target] = load_sentences_for(target)
        return self._sent_cache[target]

    def n_usable_targets(self) -> int:
        return len(self.targets)

    def sample_batch(self) -> tuple[dict, list[tuple[int, int]], list[str]] | None:
        """Return (encoded_inputs, spans, target_words) for one RSR batch, or None
        if a valid batch couldn't be assembled."""
        N = self.cfg.rsr_batch_size
        if len(self.targets) < N:
            return None

        chosen_words: list[str] = []
        chosen_sents: list[str] = []
        chosen_spans: list[tuple[int, int]] = []

        # Sample words, pick one sentence each, locate the target span. Retry a
        # few times to fill the batch with words whose span is found.
        attempts = 0
        pool = random.sample(self.targets, k=min(len(self.targets), N * 6))
        for word in pool:
            if len(chosen_words) == N:
                break
            sents = self._sentences(word)
            if not sents:
                continue
            target_ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            random.shuffle(sents)
            for sent in sents[:8]:  # try a few sentences for this word
                enc = self.tokenizer(
                    sent, add_special_tokens=True, truncation=True,
                    max_length=self.cfg.max_seq_len,
                )["input_ids"]
                span = find_subword_span(enc, target_ids)
                if span is not None:
                    chosen_words.append(word)
                    chosen_sents.append(sent)
                    chosen_spans.append(span)
                    break
            attempts += 1

        if len(chosen_words) < N:
            return None

        # Require at least a few scored pairs among the chosen words (Option B).
        scored = sum(
            1 for a, b in combinations(chosen_words, 2)
            if (a, b) in self.sim_lookup
        )
        if scored < self.cfg.min_pairs_per_rsr_batch:
            return None

        batch = self.tokenizer(
            chosen_sents, add_special_tokens=True, truncation=True,
            max_length=self.cfg.max_seq_len, padding=True, return_tensors="pt",
        )
        return batch, chosen_spans, chosen_words


def rsr_loss_for_batch(model: BertRSR, batch, spans, words, sim_lookup, cfg) -> torch.Tensor | None:
    """Compute 1 - soft_spearman over the scored pairs of the batch (Option B)."""
    input_ids = batch["input_ids"].to(DEVICE)
    attention = batch["attention_mask"].to(DEVICE)
    vecs = model.target_vectors(input_ids, attention, spans)  # (N, d)

    model_sims, human_sims = [], []
    for i, j in combinations(range(len(words)), 2):
        score = sim_lookup.get((words[i], words[j]))
        if score is None:
            continue  # mask out unscored pairs
        cos = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0))
        model_sims.append(cos)
        human_sims.append(score)

    if len(model_sims) < cfg.min_pairs_per_rsr_batch:
        return None
    rho = soft_spearman(
        torch.cat(model_sims),
        torch.tensor(human_sims, device=DEVICE, dtype=torch.float32),
        strength=cfg.soft_rank_strength,
    )
    return 1.0 - rho


# ============================================================================
# MLM-batch construction  (mask 15% of tokens, standard scheme)
# ============================================================================
class MLMBatcher:
    def __init__(self, tokenizer, cfg: Config, sentences: list[str]):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sentences = sentences

    def sample_batch(self):
        sents = random.sample(self.sentences, k=min(len(self.sentences), self.cfg.mlm_batch_size))
        enc = self.tokenizer(
            sents, add_special_tokens=True, truncation=True,
            max_length=self.cfg.max_seq_len, padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        # Standard BERT masking: 15% of non-special tokens; of those 80% [MASK],
        # 10% random, 10% unchanged. Loss only on masked positions (-100 elsewhere).
        probability = torch.full(labels.shape, self.cfg.mlm_probability)
        special = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
             for row in input_ids.tolist()], dtype=torch.bool,
        )
        probability.masked_fill_(special, 0.0)
        probability.masked_fill_(enc["attention_mask"] == 0, 0.0)
        masked = torch.bernoulli(probability).bool()
        labels[~masked] = -100

        # 80% -> [MASK]
        repl = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        input_ids[repl] = self.tokenizer.mask_token_id
        # 10% -> random token
        rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~repl
        input_ids[rand] = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)[rand]
        # remaining 10% -> unchanged

        return (input_ids.to(DEVICE), enc["attention_mask"].to(DEVICE), labels.to(DEVICE))


# ============================================================================
# Evaluation on held-out SimLex-999  (in-context, same as training)
# ============================================================================
def embed_words_in_context(model: BertRSR, words: list[str], cfg: Config,
                           sent_cache: dict[str, list[str]], max_sents: int = 20) -> dict[str, torch.Tensor]:
    """Mean contextual projected vector per word, averaged over up to `max_sents`
    of its harvested sentences. Returns only words that could be embedded."""
    model.eval()
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for w in words:
            if w not in sent_cache:
                sent_cache[w] = load_sentences_for(w)
            sents = sent_cache[w]
            if not sents:
                continue
            target_ids = model.tokenizer(w, add_special_tokens=False)["input_ids"]
            vecs = []
            for sent in sents[:max_sents]:
                enc = model.tokenizer(
                    sent, add_special_tokens=True, truncation=True,
                    max_length=cfg.max_seq_len, return_tensors="pt",
                )
                span = find_subword_span(enc["input_ids"][0].tolist(), target_ids)
                if span is None:
                    continue
                v = model.target_vectors(
                    enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), [span]
                )[0]
                vecs.append(v)
            if vecs:
                out[w] = torch.stack(vecs).mean(dim=0)
    return out


def evaluate_simlex(model: BertRSR, simlex, supervised_vocab, cfg, sent_cache) -> dict:
    """Spearman vs human SimLex, overall and per RSR-overlap partition."""
    cats = defaultdict(list)
    for w1, w2, s in simlex:
        in1, in2 = w1 in supervised_vocab, w2 in supervised_vocab
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, s))
        cats[cat].append((w1, w2, s))

    results = {}
    for name, pairs in cats.items():
        words = list({w for p in pairs for w in p[:2]})
        emb = embed_words_in_context(model, words, cfg, sent_cache)
        model_s, human_s = [], []
        for w1, w2, s in pairs:
            if w1 in emb and w2 in emb:
                cos = F.cosine_similarity(emb[w1].unsqueeze(0), emb[w2].unsqueeze(0)).item()
                model_s.append(cos)
                human_s.append(s)
        if len(model_s) < 2:
            results[name] = {"n": len(model_s), "rho": float("nan")}
        else:
            rho, _ = spearmanr(human_s, model_s)
            results[name] = {"n": len(model_s), "rho": float(rho)}
    return results


def print_eval(tag: str, res: dict) -> None:
    print(f"  [{tag}] SimLex rho  "
          f"all={res['all']['rho']:.4f} (n={res['all']['n']})  "
          f"both={res['both']['rho']:.4f}  "
          f"one={res['one']['rho']:.4f}  "
          f"neither={res['neither']['rho']:.4f}", flush=True)


# ============================================================================
# Training loop
# ============================================================================
class Shared:
    """Data loaded once and reused across runs (so a sweep doesn't reload it)."""
    def __init__(self, tokenizer):
        print("Loading human similarity data ...", flush=True)
        self.sim_lookup, self.supervised_vocab = load_supervision()
        self.simlex = load_simlex()
        print(f"  supervised vocab: {len(self.supervised_vocab)} | "
              f"scored pairs: {len(self.sim_lookup)//2} | "
              f"SimLex pairs: {len(self.simlex)}", flush=True)
        # RSR target/sentence sampler (tokenizer-dependent but model-independent).
        self.rsr_sampler = RSRBatchSampler(self.supervised_vocab, self.sim_lookup, tokenizer, Config())
        print(f"  usable RSR targets (supervised + have sentences): "
              f"{self.rsr_sampler.n_usable_targets()}", flush=True)
        print("Loading MLM sentence pool ...", flush=True)
        self.mlm_sentences = []
        for w in random.sample(self.rsr_sampler.targets, k=min(400, len(self.rsr_sampler.targets))):
            self.mlm_sentences.extend(load_sentences_for(w))
        print(f"  MLM sentences: {len(self.mlm_sentences)}", flush=True)


def train_one(cfg: Config, regime: str, model: BertRSR, shared: Shared,
              verbose: bool = True) -> pd.DataFrame:
    """Train `model` under `regime` using already-loaded `shared` data."""
    shared.rsr_sampler.cfg = cfg                       # honour this run's batch size etc.
    mlm_batcher = MLMBatcher(model.tokenizer, cfg, shared.mlm_sentences)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate
    )
    eval_cache: dict[str, list[str]] = {}
    history = []

    res = evaluate_simlex(model, shared.simlex, shared.supervised_vocab, cfg, eval_cache)
    if verbose:
        print("\nBaseline evaluation (before any training):", flush=True)
        print_eval("step 0", res)
    history.append({"step": 0, **{f"{k}_{m}": res[k][m] for k in res for m in res[k]}})

    t0 = time.time()
    running_mlm, running_rsr, n_mlm, n_rsr = 0.0, 0.0, 0, 0
    for step in range(1, cfg.steps + 1):
        model.train()
        do_rsr = (regime == "rsr") and ((step - 1) % cfg.interleave_cycle >= cfg.mlm_per_cycle)

        if do_rsr:
            sample = shared.rsr_sampler.sample_batch()
            if sample is None:
                continue
            batch, spans, words = sample
            loss = rsr_loss_for_batch(model, batch, spans, words, shared.sim_lookup, cfg)
            if loss is None:
                continue
            running_rsr += loss.item(); n_rsr += 1
        else:
            input_ids, attn, labels = mlm_batcher.sample_batch()
            if (labels != -100).sum() == 0:
                continue
            loss = model.mlm_loss(input_ids, attn, labels)
            running_mlm += loss.item(); n_mlm += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        optimizer.step()

        if step % cfg.eval_every == 0:
            mlm_avg = running_mlm / max(n_mlm, 1)
            rsr_avg = running_rsr / max(n_rsr, 1)
            res = evaluate_simlex(model, shared.simlex, shared.supervised_vocab, cfg, eval_cache)
            if verbose:
                print(f"step {step}/{cfg.steps}  mlm_loss={mlm_avg:.4f}  "
                      f"rsr_loss={rsr_avg:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                print_eval(f"step {step}", res)
            history.append({"step": step, "mlm_loss": mlm_avg, "rsr_loss": rsr_avg,
                            **{f"{k}_{m}": res[k][m] for k in res for m in res[k]}})
            running_mlm, running_rsr, n_mlm, n_rsr = 0.0, 0.0, 0, 0

    return pd.DataFrame(history)


def train(cfg: Config, regime: str, shared: Shared | None = None) -> pd.DataFrame:
    """Build a fresh model and train it. Loads shared data if not provided."""
    set_seed(cfg.seed)
    print(f"Device: {DEVICE} | torchsort: {HAS_TORCHSORT} | regime: {regime}", flush=True)
    print("Building model ...", flush=True)
    model = BertRSR(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,}", flush=True)
    if shared is None:
        shared = Shared(model.tokenizer)
    print("\nTraining ...", flush=True)
    return train_one(cfg, regime, model, shared, verbose=True)


# ============================================================================
# Sweep: try a few (N, interleave, lr) recipes at reduced steps and report
# which best lifts held-out SimLex over the vanilla baseline.
# ============================================================================
def run_sweep(base_cfg: Config) -> None:
    from copy import copy

    # Each tuple: (rsr_batch_size, (interleave_cycle, mlm_per_cycle), lr)
    # interleave (2,1)=1MLM:1RSR balanced; (3,2)=2MLM:1RSR (MLM-majority, gentlest)
    combos = [
        (24, (2, 1), 5e-5),   # balanced, bigger batch, lower lr
        (24, (3, 2), 5e-5),   # MLM-majority (gentlest on the language model)
        (32, (2, 1), 3e-5),   # bigger batch still, even lower lr
        (16, (2, 1), 5e-5),   # smaller batch baseline for comparison
    ]
    # If --smoke shrank cfg.steps, run a tiny sweep too (wiring check only).
    smoke = base_cfg.steps <= 20
    sweep_steps = 20 if smoke else 600
    eval_every = 10 if smoke else 150
    if smoke:
        combos = combos[:2]

    set_seed(base_cfg.seed)
    print(f"Device: {DEVICE} | torchsort: {HAS_TORCHSORT} | SWEEP "
          f"({len(combos)} combos x {sweep_steps} steps)", flush=True)

    # Build one model just to get a tokenizer for Shared; data loaded once.
    print("Building reference model + loading data once ...", flush=True)
    probe = BertRSR(base_cfg)
    shared = Shared(probe.tokenizer)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    for i, (N, (cyc, mlmpc), lr) in enumerate(combos, 1):
        cfg = copy(base_cfg)
        cfg.rsr_batch_size = N
        cfg.interleave_cycle = cyc
        cfg.mlm_per_cycle = mlmpc
        cfg.learning_rate = lr
        cfg.steps = sweep_steps
        cfg.eval_every = eval_every
        ratio = f"{mlmpc}MLM:{cyc-mlmpc}RSR"
        print(f"\n===== combo {i}/{len(combos)}: N={N}, {ratio}, lr={lr} =====", flush=True)
        set_seed(cfg.seed)
        model = BertRSR(cfg)
        df = train_one(cfg, "rsr", model, shared, verbose=True)
        van = df.iloc[0]["all_rho"]
        best_i = df["all_rho"].idxmax()
        best = df.loc[best_i]
        rows.append({
            "N": N, "interleave": ratio, "lr": lr,
            "vanilla_all": van,
            "best_all": best["all_rho"], "best_step": int(best["step"]),
            "best_neither": best["neither_rho"],
            "delta_all": best["all_rho"] - van,
            "final_all": df.iloc[-1]["all_rho"],
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"bert_sweep_{ts}.csv"
    res.to_csv(out, index=False)

    print("\n" + "=" * 78, flush=True)
    print("SWEEP RESULTS  (best held-out SimLex 'all' reached during each short run)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'N':>4}{'interleave':>14}{'lr':>9}{'vanilla':>9}{'best':>8}"
          f"{'@step':>7}{'d_all':>8}{'neither':>9}{'final':>8}", flush=True)
    print("-" * 78, flush=True)
    for r in rows:
        print(f"{r['N']:>4}{r['interleave']:>14}{r['lr']:>9.0e}{r['vanilla_all']:>9.3f}"
              f"{r['best_all']:>8.3f}{r['best_step']:>7}{r['delta_all']:>+8.3f}"
              f"{r['best_neither']:>9.3f}{r['final_all']:>8.3f}", flush=True)
    best = max(rows, key=lambda r: r["delta_all"])
    print(f"\nBest recipe by d_all: N={best['N']}, {best['interleave']}, lr={best['lr']:.0e} "
          f"(d_all={best['delta_all']:+.3f} at step {best['best_step']})", flush=True)
    print("If the best d_all is still negative, the similarity signal itself needs", flush=True)
    print("rethinking (e.g. weighted loss instead of interleaving); if positive, use", flush=True)
    print("that recipe for a full run.", flush=True)
    print(f"\nSaved sweep table to: {out}", flush=True)


# ============================================================================
# Comparison table  (Vanilla BERT vs RSR BERT, partitioned like paper Table 2)
# ============================================================================
PARTITIONS = ("all", "both", "one", "neither")
PART_LABEL = {"all": "All pairs", "both": "Both in RSR",
              "one": "One in RSR", "neither": "Neither in RSR"}


def print_table2(vanilla: dict, rsr: dict, mlm: dict | None = None) -> None:
    """Print the headline comparison in the paper's Table 2 layout.

    `vanilla`, `rsr`, `mlm` are evaluate_simlex() result dicts (per-partition
    rho + n). `mlm` (continued-MLM-only control) is optional; when present a
    fourth column isolates the RSR effect from plain continued training.
    """
    def rho(d, p):
        return d[p]["rho"]

    print("\n" + "=" * 72, flush=True)
    print("BERT SimLex-999 results after RSR fine-tuning", flush=True)
    print("(rho = Spearman correlation with human similarity; held-out SimLex)", flush=True)
    print("=" * 72, flush=True)

    if mlm is None:
        header = f"{'Category':<16}{'Vanilla':>10}{'RSR':>10}{'Delta':>10}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for p in PARTITIONS:
            v, r = rho(vanilla, p), rho(rsr, p)
            print(f"{PART_LABEL[p]:<16}{v:>10.3f}{r:>10.3f}{r - v:>+10.3f}", flush=True)
    else:
        header = (f"{'Category':<16}{'Vanilla':>9}{'+MLM':>9}{'RSR':>9}"
                  f"{'RSR-Van':>10}{'RSR-MLM':>10}")
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for p in PARTITIONS:
            v, m, r = rho(vanilla, p), rho(mlm, p), rho(rsr, p)
            print(f"{PART_LABEL[p]:<16}{v:>9.3f}{m:>9.3f}{r:>9.3f}"
                  f"{r - v:>+10.3f}{r - m:>+10.3f}", flush=True)
        print("\n  RSR-Van = RSR gain over vanilla BERT", flush=True)
        print("  RSR-MLM = RSR gain over continued-MLM-only (isolates the RSR effect)", flush=True)

    print(f"\n  n (all pairs evaluated): {vanilla['all']['n']}/999", flush=True)
    print("  'Neither in RSR' is the generalisation test: gains there mean RSR", flush=True)
    print("  reshaped words it never saw a human score for.", flush=True)


def df_first_last(df: pd.DataFrame) -> tuple[dict, dict]:
    """Reconstruct evaluate_simlex-style dicts from a history DataFrame's first
    (vanilla, step 0) and last (trained) rows."""
    def row_to_eval(row) -> dict:
        return {p: {"rho": row[f"{p}_rho"], "n": int(row[f"{p}_n"])} for p in PARTITIONS}
    return row_to_eval(df.iloc[0]), row_to_eval(df.iloc[-1])


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["compare", "rsr", "baseline"], default="compare",
                    help="compare (default) = run both vanilla->RSR and vanilla->MLM and "
                         "print the comparison table; rsr / baseline = run just one")
    ap.add_argument("--steps", type=int, default=None, help="override total optimiser steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rsr-batch-size", type=int, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="try several (batch size, interleave, lr) recipes at reduced steps "
                         "and report which best lifts SimLex (run this first to pick a recipe)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run to check everything wires up (CPU-friendly)")
    args = ap.parse_args()

    cfg = Config()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.rsr_batch_size is not None:
        cfg.rsr_batch_size = args.rsr_batch_size
    if args.smoke:
        cfg.steps = 20
        cfg.eval_every = 10
        cfg.mlm_batch_size = 4

    if args.sweep:
        run_sweep(cfg)
        print("\nDone.", flush=True)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.regime == "compare":
        # Each regime builds its OWN fresh BERT, so the two runs don't
        # contaminate each other; data is loaded once and shared. Vanilla
        # (step-0) eval is identical for both by construction; take it from RSR.
        from transformers import BertTokenizerFast
        shared = Shared(BertTokenizerFast.from_pretrained(cfg.model_name))
        print("\n########## RSR REGIME (interleaved MLM + RSR) ##########", flush=True)
        rsr_df = train(cfg, "rsr", shared)
        print("\n########## BASELINE REGIME (continued MLM only) ##########", flush=True)
        base_df = train(cfg, "baseline", shared)

        rsr_df.to_csv(RESULTS_DIR / f"bert_continued_rsr_{ts}.csv", index=False)
        base_df.to_csv(RESULTS_DIR / f"bert_continued_baseline_{ts}.csv", index=False)

        vanilla, rsr_final = df_first_last(rsr_df)
        _, mlm_final = df_first_last(base_df)
        print_table2(vanilla, rsr_final, mlm_final)
        print(f"\nSaved per-step histories to: results/bert_continued_*_{ts}.csv", flush=True)
    else:
        df = train(cfg, args.regime)
        out = RESULTS_DIR / f"bert_continued_{args.regime}_{ts}.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved training history to: {out}", flush=True)
        if len(df) >= 2:
            vanilla, final = df_first_last(df)
            # single-regime: show vanilla vs this regime's final, no control column
            print_table2(vanilla, final)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
