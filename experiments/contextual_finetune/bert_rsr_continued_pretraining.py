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

By default NOTHING is frozen and there is NO projection head: the RSR loss is
applied directly to the last encoder layer's 768-d states and gradients flow
through the whole backbone (supervisor's request, 4 Jun - "we want to be able
to update the conceptual representations in the backbone model"). Mark
Ormerod's original recipe (freeze embeddings + layers 0-10, train only the
final block + a 128-d projection head) is still available via
--frozen-layers 11 --proj-dim 128.

--------------------------------------------------------------------------------
DATA (reuses what the repo already has)
--------------------------------------------------------------------------------
  * Human similarity (RSR supervision + ground-truth RDM):
        SimVerb-3500 + THINGS,  pooled & min-max normalised.
        MEN is EXCLUDED by default (supervisor, 4 Jun: MEN subjects rated
        relatedness, not strict similarity). Restore it with --use-men.
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
  * Interleave MLM and RSR as separate batches (supervisor's request), balanced
    1:1 per cycle; --combine weighted mixes both losses in every batch instead.
  * Supervision = SimVerb-3500 + THINGS. MEN excluded by default (relatedness,
    not similarity - supervisor, 4 Jun); restore with --use-men.
  * Nothing frozen, no projection head by default (supervisor, 4 Jun): the RSR
    loss is applied to the last encoder layer's raw states and the whole
    backbone trains. Mark's recipe = --frozen-layers 11 --proj-dim 128.
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

import os

# Run fully offline: use the locally cached BERT and NEVER touch the network.
# Without this, transformers pings huggingface.co on every from_pretrained() to
# check for updates - so a dropped connection mid-run crashes the whole job
# (which is exactly what killed an overnight sweep). Must be set before the
# transformers import. The model is already in the local HF cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
# Configuration
# Mark Ormerod's original = --frozen-layers 11 --proj-dim 128 --use-men)
# ============================================================================
class Config:
    model_name = "bert-base-uncased"
    num_frozen_layers = 0
    projection_dim = 0
    use_men = False
    # Which hidden layer the RSR target vectors are read from. None = the last
    # hidden state (= layer 12), the original default. 0..12 picks a specific
    # layer (0 = embedding output). Motivated by Vulic et al. 2020: lexical-
    # semantic content peaks in the MIDDLE layers and degrades toward the top,
    # so the last layer may be a poor read-out point. Evaluation reads the same
    # layer as training, so comparisons stay honest.
    rsr_layer: int | None = None
    # Train ONLY this encoder layer (freeze embeddings + every other encoder
    # layer; the MLM head stays trainable so the anchor loss can still adapt).
    # None = freezing is governed by num_frozen_layers as usual. Used by the
    # layer sweep to ask which single layer best absorbs the RSR constraint.
    train_only_layer: int | None = None
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
    eval_every = 100
    seed = 1

    # Interleaving (rsr regime): alternate MLM and RSR batches in a repeating
    # cycle of length `interleave_cycle`; the first `mlm_per_cycle` steps of each
    # cycle are MLM, the rest RSR. Default 1 MLM : 1 RSR (balanced).
    #
    # NB: the first experiment used 1:2 (RSR-majority) reasoning from Mark's
    # lambda=0.9 - but Mark's weighting kept MLM present in *every* batch, whereas
    # interleaving with 1:2 leaves 2 of 3 batches with NO language signal. That
    # starved BERT and degraded SimLex. Balanced is the safer default; revisit
    # the ratio once a stable recipe is found.
    interleave_cycle = 2
    mlm_per_cycle = 1

    # How the RSR regime combines the two signals:
    #   "interleave" -> alternate pure MLM and pure RSR batches (above settings).
    #                   sometimes prone to overshoot because the
    #                   pure-RSR batches have nothing anchoring the geometry.
    #   "weighted"   -> Mark's approach: every batch does BOTH at once,
    #                   loss = (1 - rsr_lambda) * MLM + rsr_lambda * RSR.
    #                   MLM is always present, which tends to stop RSR running away.
    combine = "interleave"
    rsr_lambda = 0.5     # weight on the RSR term in "weighted" mode (Mark used 0.9)

    # Early stopping: track held-out SimLex at each eval and keep the best
    # checkpoint, so a run reports its peak rather than an overshot final model.
    early_stop = True
    early_stop_metric = "all"   # which SimLex partition to track ("all"/"neither")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Differentiable Spearman loss  (the RSR term is  1 - rho_soft)
# ----------------------------------------------------------------------------
# THE IDEA. RSR wants the model's word-pair similarities to *rank* in the same
# order as the human ratings (pair A more similar than pair B than pair C ...).
# Rank agreement is exactly Spearman's rho. But Spearman needs each value's
# RANK, and ranking = sorting, which is a step function: its gradient is zero
# almost everywhere, so you cannot backpropagate a training signal through it.
# The fix is a SOFT (smooth, differentiable) approximation of the rank, so the
# whole "1 - rho" quantity has a usable gradient and can drive training.
# ============================================================================
def soft_rank_custom(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """A differentiable approximation of 'what rank does each value hold?'.

    Exact rank of x[i] = 1 + (number of other values smaller than it). The
    'number smaller' is a hard count (non-differentiable). We soften it: for
    every pair (i, j), sigmoid(x[i] - x[j]) is ~1 when x[i] >> x[j], ~0 when
    x[i] << x[j], and 0.5 at a tie. Summing that soft count over j gives a
    smooth stand-in for the rank that *does* have a gradient. `strength`
    sharpens the sigmoid toward a true step (higher = closer to a hard rank).
    Only used when the faster `torchsort` library is not installed.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    diff = x.unsqueeze(2) - x.unsqueeze(1)          # diff[.,i,j] = x[i] - x[j]
    ranks = torch.sigmoid(diff * strength).sum(dim=2) + 0.5   # soft "# below" + offset
    return ranks.squeeze(0)


def soft_spearman(pred: torch.Tensor, target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Differentiable Spearman correlation between two 1-D score vectors.

    Spearman = Pearson correlation computed on RANKS instead of raw values. So:
    soft-rank both inputs, then take the ordinary (differentiable) Pearson
    correlation of those soft ranks. Returns a value in roughly [-1, 1]; the
    RSR loss is 1 - this, so minimising the loss == maximising rank agreement.
    `pred` = the model's pair similarities, `target` = the human scores.
    """
    if HAS_TORCHSORT:
        # torchsort.soft_rank: a faster, better-conditioned soft rank (same idea).
        pred_r = torchsort.soft_rank(pred.unsqueeze(0), regularization_strength=strength).squeeze(0)
        tgt_r = torchsort.soft_rank(target.unsqueeze(0), regularization_strength=strength).squeeze(0)
    else:
        pred_r = soft_rank_custom(pred, strength)
        tgt_r = soft_rank_custom(target, strength)
    # Pearson correlation of the two rank vectors: centre each, then
    # cov / (std * std). The 1e-8 guards against divide-by-zero on a flat batch.
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


def load_supervision(use_men: bool = False) -> tuple[dict[tuple[str, str], float], set[str]]:
    """Return {(w1,w2)->score} lookup (both orderings) and the supervised vocab.

    MEN is excluded unless `use_men` (it measures relatedness, not strict
    similarity). Words are normalised to the cache/sentence convention:
    lowercase, MWEs as space-separated tokens (THINGS underscores -> spaces).
    """
    all_pairs = _normalize(load_simverb()) + _normalize(load_things())
    if use_men:
        all_pairs += _normalize(load_men())

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
# Model:  BERT-for-MLM
# ============================================================================
class BertRSR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
        # BertForMaskedLM gives us both the encoder (.bert) and the MLM head (.cls).
        self.mlm = BertForMaskedLM.from_pretrained(cfg.model_name)
        hidden = self.mlm.config.hidden_size
        # projection_dim == 0 (default): no head, the RSR loss acts directly on
        # the last-layer 768-d states so the constraint must reshape the
        # backbone itself rather than a layer on top.
        self.projection = (
            nn.Linear(hidden, cfg.projection_dim) if cfg.projection_dim > 0
            else nn.Identity()
        )

        # --- optionally freeze embeddings + first N encoder layers ---
        # Default num_frozen_layers=0: nothing frozen, the whole backbone
        # trains. 11 = Mark's recipe (only the final block + heads train).
        if cfg.num_frozen_layers > 0:
            for p in self.mlm.bert.embeddings.parameters():
                p.requires_grad = False
            for i in range(cfg.num_frozen_layers):
                for p in self.mlm.bert.encoder.layer[i].parameters():
                    p.requires_grad = False

        # --- or train ONE encoder layer only (layer-localisation experiment) --
        # Freeze embeddings + every encoder layer except train_only_layer.
        # The MLM head (.cls) stays trainable so the anchor loss can adapt.
        # NB: encoder.layer is 0-indexed, so "layer k" here = encoder.layer[k-1]
        # to match the 1..12 numbering of hidden_states (see target_vectors).
        if cfg.train_only_layer is not None:
            k = cfg.train_only_layer
            assert 1 <= k <= len(self.mlm.bert.encoder.layer), \
                f"train_only_layer must be 1..{len(self.mlm.bert.encoder.layer)}"
            for p in self.mlm.bert.embeddings.parameters():
                p.requires_grad = False
            for i, block in enumerate(self.mlm.bert.encoder.layer):
                if i != k - 1:
                    for p in block.parameters():
                        p.requires_grad = False

        self.to(DEVICE)

    # ---- (a) MLM batch ----------------------------------------------------
    def mlm_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        out = self.mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    # ---- (b) RSR batch: one contextual vector per target word -------------
    def target_vectors(self, input_ids, attention_mask, spans) -> torch.Tensor:
        """Turn N sentences into N word vectors: one per target word, taken
        from that word *in its sentence* (BERT has no context-free embedding).

        For each row we run BERT, then average the last-layer vectors of just
        the target word's sub-word tokens (`spans` says which positions those
        are). A rare word like 'aardvark' is split into several word-pieces, so
        we mean-pool them back into a single vector. `self.projection` is a
        128-d Linear in Mark's recipe but nn.Identity by default (projection_dim
        = 0), so by default this returns BERT's own raw 768-d states and the RSR
        loss must reshape the backbone itself. Returns (N, 768 or proj_dim).
        """
        # Read from the configured layer. hidden_states is a 13-tuple:
        # [0] = embedding output, [1..12] = each encoder layer's output, so
        # hidden_states[12] == last_hidden_state. rsr_layer=None keeps the
        # original behaviour (last layer); rsr_layer=k reads layer k instead
        # (Vulic et al. 2020: middle layers carry the most lexical semantics).
        if self.cfg.rsr_layer is None:
            out = self.mlm.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (N, L, H): N sentences, L tokens, H=768
        else:
            out = self.mlm.bert(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
            hidden = out.hidden_states[self.cfg.rsr_layer]
        vecs = []
        for j, (start, end) in enumerate(spans):
            # Clamp the span into the real (non-padding) length, just in case a
            # span ran past where this row was truncated.
            seq_len = int(attention_mask[j].sum().item())
            end = min(end, seq_len)
            start = min(start, max(seq_len - 1, 0))
            vecs.append(hidden[j, start:end].mean(dim=0))   # mean-pool sub-words
        stacked = torch.stack(vecs, dim=0)
        return self.projection(stacked)


# ============================================================================
# RSR-batch construction  (Option B: any words; mask unscored pairs)
# ============================================================================
class RSRBatchSampler:
    """Builds RSR batches of N (sentence, target) items where the targets have
    cached sentences."""

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
    """The RSR loss for one batch: align the model's similarity ordering to the
    human one. This is the method in three steps-

    Step 1: get one contextual vector per target word (target_vectors).
    Step 2: for every one of the N(N-1)/2 word pairs, the model's similarity is
            the cosine of their two vectors, and the human similarity is looked
            up. Pairs with NO human score are dropped -- this is "Option B": we
            include any words we have sentences for, and simply mask the loss to
            whichever pairs happen to have a human rating.
    Step 3: feed the two parallel lists (model cosines, human scores) to
            soft_spearman and return 1 - rho. Returns None if too few scored
            pairs survived to give a stable gradient.
    """
    input_ids = batch["input_ids"].to(DEVICE)
    attention = batch["attention_mask"].to(DEVICE)
    vecs = model.target_vectors(input_ids, attention, spans)  # (N, d), one row per word

    # Build the two aligned vectors: model_sims[k] and human_sims[k] are the
    # same pair, so soft_spearman compares like with like.
    model_sims, human_sims = [], []
    for i, j in combinations(range(len(words)), 2):
        score = sim_lookup.get((words[i], words[j]))
        if score is None:
            continue  # no human rating for this pair -> mask it out (Option B)
        cos = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0))
        model_sims.append(cos)
        human_sims.append(score)

    if len(model_sims) < cfg.min_pairs_per_rsr_batch:
        return None   # too few scored pairs this batch -> skip it
    rho = soft_spearman(
        torch.cat(model_sims),                                       # keeps the graph (trainable)
        torch.tensor(human_sims, device=DEVICE, dtype=torch.float32),  # constant targets
        strength=cfg.soft_rank_strength,
    )
    return 1.0 - rho   # 0 when the orderings match perfectly, up to ~2 when reversed


# ============================================================================
# MLM-batch construction  (mask 15% of tokens, standard scheme)
# ============================================================================
class MLMBatcher:
    def __init__(self, tokenizer, cfg: Config, sentences: list[str]):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sentences = sentences

    def sample_batch(self):
        # `labels` starts as a copy of the true tokens; we then corrupt some
        # positions in `input_ids` and tell the loss to predict the originals
        # only at those positions (everything else is set to -100 = "ignore").
        sents = random.sample(self.sentences, k=min(len(self.sentences), self.cfg.mlm_batch_size))
        enc = self.tokenizer(
            sents, add_special_tokens=True, truncation=True,
            max_length=self.cfg.max_seq_len, padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        # Standard BERT masking (Devlin et al. 2019), unchanged from Mark's code:
        # pick 15% of real tokens to predict; of those, 80% become [MASK], 10%
        # become a random token, 10% are left as-is. The 10% random / 10% kept
        # stop BERT from only ever learning to fill literal [MASK] slots.
        probability = torch.full(labels.shape, self.cfg.mlm_probability)   # 0.15 everywhere
        # Never mask special tokens ([CLS]/[SEP]) or padding -> set their prob to 0.
        special = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
             for row in input_ids.tolist()], dtype=torch.bool,
        )
        probability.masked_fill_(special, 0.0)
        probability.masked_fill_(enc["attention_mask"] == 0, 0.0)
        masked = torch.bernoulli(probability).bool()   # the chosen 15% of positions
        labels[~masked] = -100                          # loss ignores everything not chosen

        # Of the chosen positions: 80% -> [MASK]
        repl = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        input_ids[repl] = self.tokenizer.mask_token_id
        # of the remaining 20%, half (=10% overall) -> a random token
        rand = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~repl
        input_ids[rand] = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)[rand]
        # the last 10% are left unchanged (input_ids untouched, but still scored)

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
    """Spearman vs human SimLex, overall and split by RSR-supervision overlap.

    The split is the whole point of the evaluation. For each held-out SimLex
    pair we ask how many of its two words appeared ANYWHERE in the RSR
    supervision vocabulary (never as this pair -- SimLex itself is never
    trained on):
        both    -> both words were supervised (with other partners)
        one     -> exactly one was
        neither -> NEITHER was ever given a human similarity score
    'neither' is the generalisation test: improvement there can't be the model
    memorising supervised words, so it shows RSR reshaped the space as a whole.
    """
    cats = defaultdict(list)
    for w1, w2, s in simlex:
        in1, in2 = w1 in supervised_vocab, w2 in supervised_vocab
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, s))   # every pair also counts toward "all"
        cats[cat].append((w1, w2, s))

    results = {}
    for name, pairs in cats.items():
        # Embed each word in context, then score each pair by cosine similarity
        # and correlate the model's similarities with the human SimLex scores.
        words = list({w for p in pairs for w in p[:2]})
        emb = embed_words_in_context(model, words, cfg, sent_cache)
        model_s, human_s = [], []
        for w1, w2, s in pairs:
            if w1 in emb and w2 in emb:   # skip pairs we couldn't embed (no sentences)
                cos = F.cosine_similarity(emb[w1].unsqueeze(0), emb[w2].unsqueeze(0)).item()
                model_s.append(cos)
                human_s.append(s)
        if len(model_s) < 2:
            results[name] = {"n": len(model_s), "rho": float("nan")}
        else:
            # Hard (real) Spearman here -- eval needs no gradient, so we use the
            # exact scipy version rather than the soft approximation.
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
# Layer probe  (--probe-layers): which layer of VANILLA BERT is already best
# at SimLex?  No training. Each sentence is forwarded ONCE with
# output_hidden_states=True, and the target span is pooled from every layer,
# so the whole 13-layer curve costs about the same as one normal evaluation.
# Motivation: Vulic et al. 2020 found lexical semantics peaks mid-stack; if a
# middle layer beats layer 12 here, it is both a better vanilla baseline and a
# better place to attach the RSR loss (--rsr-layer).
# ============================================================================
def probe_layers(cfg: Config, max_sents: int = 20) -> pd.DataFrame:
    set_seed(cfg.seed)
    print(f"Device: {DEVICE} | LAYER PROBE (vanilla {cfg.model_name}, "
          f"no training, {max_sents} sentences/word)", flush=True)
    model = BertRSR(cfg)
    model.eval()
    shared = Shared(model.tokenizer, cfg)
    n_layers = model.mlm.config.num_hidden_layers  # 12 for bert-base

    # Partition the SimLex pairs once (same logic as evaluate_simlex).
    cats = defaultdict(list)
    for w1, w2, s in shared.simlex:
        in1, in2 = w1 in shared.supervised_vocab, w2 in shared.supervised_vocab
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, s))
        cats[cat].append((w1, w2, s))

    # One pass over all SimLex words: accumulate a per-layer mean vector each.
    words = sorted({w for p in shared.simlex for w in p[:2]})
    emb: list[dict[str, torch.Tensor]] = [dict() for _ in range(n_layers + 1)]
    t0 = time.time()
    with torch.no_grad():
        for wi, w in enumerate(words):
            sents = load_sentences_for(w)
            if not sents:
                continue
            target_ids = model.tokenizer(w, add_special_tokens=False)["input_ids"]
            sums = None  # (13, H) running sum of this word's span vectors
            count = 0
            for sent in sents[:max_sents]:
                enc = model.tokenizer(sent, add_special_tokens=True, truncation=True,
                                      max_length=cfg.max_seq_len, return_tensors="pt")
                span = find_subword_span(enc["input_ids"][0].tolist(), target_ids)
                if span is None:
                    continue
                out = model.mlm.bert(input_ids=enc["input_ids"].to(DEVICE),
                                     attention_mask=enc["attention_mask"].to(DEVICE),
                                     output_hidden_states=True)
                # hidden_states: tuple of 13 (1, L, H) tensors -> stack to (13, L, H)
                hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # (13, L, H)
                start, end = span
                vec = hs[:, start:end].mean(dim=1)  # (13, H) span mean per layer
                sums = vec if sums is None else sums + vec
                count += 1
            if count:
                mean = sums / count
                for L in range(n_layers + 1):
                    emb[L][w] = mean[L]
            if (wi + 1) % 200 == 0:
                print(f"  embedded {wi + 1}/{len(words)} words "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # Score every layer on every partition.
    rows = []
    for L in range(n_layers + 1):
        row = {"layer": L}
        for name, pairs in cats.items():
            model_s, human_s = [], []
            for w1, w2, s in pairs:
                if w1 in emb[L] and w2 in emb[L]:
                    cos = F.cosine_similarity(emb[L][w1].unsqueeze(0),
                                              emb[L][w2].unsqueeze(0)).item()
                    model_s.append(cos)
                    human_s.append(s)
            rho, _ = spearmanr(human_s, model_s)
            row[f"{name}_rho"] = float(rho)
            row[f"{name}_n"] = len(model_s)
        rows.append(row)
    df = pd.DataFrame(rows)

    print("\n" + "=" * 62, flush=True)
    print("LAYER PROBE  (vanilla SimLex rho per hidden layer; 0 = embeddings)", flush=True)
    print("=" * 62, flush=True)
    print(f"{'layer':>5}{'all':>10}{'both':>10}{'one':>10}{'neither':>10}", flush=True)
    best = df["all_rho"].idxmax()
    for _, r in df.iterrows():
        mark = "  <-- best (all)" if r["layer"] == df.loc[best, "layer"] else ""
        print(f"{int(r['layer']):>5}{r['all_rho']:>10.4f}{r['both_rho']:>10.4f}"
              f"{r['one_rho']:>10.4f}{r['neither_rho']:>10.4f}{mark}", flush=True)
    print(f"\nLast layer (12) all-pairs rho: {df.iloc[-1]['all_rho']:.4f} | "
          f"best layer {int(df.loc[best, 'layer'])}: {df.loc[best, 'all_rho']:.4f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"bert_layer_probe_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"Saved layer probe to: {out}", flush=True)
    return df


# ============================================================================
# Training loop
# ============================================================================
class Shared:
    """Data loaded once and reused across runs (so a sweep doesn't reload it)."""
    def __init__(self, tokenizer, cfg: Config | None = None):
        cfg = cfg or Config()
        print(f"Loading human similarity data (MEN {'IN' if cfg.use_men else 'EXCLUDED'}) ...",
              flush=True)
        self.sim_lookup, self.supervised_vocab = load_supervision(use_men=cfg.use_men)
        self.simlex = load_simlex()
        print(f"  supervised vocab: {len(self.supervised_vocab)} | "
              f"scored pairs: {len(self.sim_lookup)//2} | "
              f"SimLex pairs: {len(self.simlex)}", flush=True)
        # RSR target/sentence sampler (tokenizer-dependent but model-independent).
        self.rsr_sampler = RSRBatchSampler(self.supervised_vocab, self.sim_lookup, tokenizer, cfg)
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

    def do_rsr_loss():
        """Sample an RSR batch and return its loss (or None if unusable)."""
        sample = shared.rsr_sampler.sample_batch()
        if sample is None:
            return None
        batch, spans, words = sample
        return rsr_loss_for_batch(model, batch, spans, words, shared.sim_lookup, cfg)

    def do_mlm_loss():
        """Sample an MLM batch and return its loss (or None if no masked tokens)."""
        input_ids, attn, labels = mlm_batcher.sample_batch()
        if (labels != -100).sum() == 0:
            return None
        return model.mlm_loss(input_ids, attn, labels)

    res = evaluate_simlex(model, shared.simlex, shared.supervised_vocab, cfg, eval_cache)
    if verbose:
        print("\nBaseline evaluation (before any training):", flush=True)
        print_eval("step 0", res)
    history.append({"step": 0, **{f"{k}_{m}": res[k][m] for k in res for m in res[k]}})

    # Early-stopping bookkeeping: remember the best-SimLex state.
    best_metric = res[cfg.early_stop_metric]["rho"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} \
        if cfg.early_stop else None
    best_step = 0

    # Three training modes share this one loop:
    #   baseline regime  -> MLM only (the control: continued pre-training, no RSR)
    #   rsr + weighted   -> every step does BOTH losses, combined by rsr_lambda
    #   rsr + interleave -> each step is EITHER a pure MLM or a pure RSR step
    weighted = (regime == "rsr") and (cfg.combine == "weighted")

    t0 = time.time()
    # Running sums so the periodic print can show the average loss of each kind.
    running_mlm, running_rsr, n_mlm, n_rsr = 0.0, 0.0, 0, 0
    for step in range(1, cfg.steps + 1):
        model.train()

        if regime != "rsr":
            # CONTROL: masked-language-modelling only, no similarity signal.
            loss = do_mlm_loss()
            if loss is None:
                continue
            running_mlm += loss.item(); n_mlm += 1

        elif weighted:
            # WEIGHTED (Mark's approach, our default): both losses every step,
            # so MLM is always present to anchor the language model while RSR
            # reshapes the geometry. lambda is the weight on the RSR term:
            #   loss = (1 - lambda) * MLM  +  lambda * RSR.
            # Keeping MLM in every batch is what stops RSR "running away" (the
            # pure-RSR interleave overshoots because some steps have no anchor).
            mlm_loss = do_mlm_loss()
            rsr_loss = do_rsr_loss()
            if mlm_loss is None or rsr_loss is None:
                continue
            loss = (1.0 - cfg.rsr_lambda) * mlm_loss + cfg.rsr_lambda * rsr_loss
            running_mlm += mlm_loss.item(); n_mlm += 1
            running_rsr += rsr_loss.item(); n_rsr += 1

        else:
            # INTERLEAVE: alternate pure MLM and pure RSR batches on a fixed
            # cycle. With cycle=2, mlm_per_cycle=1 this is 1 MLM : 1 RSR.
            do_rsr = (step - 1) % cfg.interleave_cycle >= cfg.mlm_per_cycle
            if do_rsr:
                loss = do_rsr_loss()
                if loss is None:
                    continue
                running_rsr += loss.item(); n_rsr += 1
            else:
                loss = do_mlm_loss()
                if loss is None:
                    continue
                running_mlm += loss.item(); n_mlm += 1

        # One optimiser update. clip_grad_norm caps the global gradient norm at
        # 1.0 so a single noisy RSR batch can't take a huge, destabilising step.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        optimizer.step()

        # Every eval_every steps (100, per supervisor): measure held-out SimLex
        # so we can SEE the training dynamics, log the row, and track the peak.
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

            # EARLY STOPPING. RSR overshoots: SimLex climbs, peaks, then decays
            # if you keep training. So whenever this eval beats the best so far,
            # snapshot the weights. At the end we roll back to that snapshot, so
            # the reported model is the peak rather than the (worse) final one.
            if cfg.early_stop and res[cfg.early_stop_metric]["rho"] > best_metric:
                best_metric = res[cfg.early_stop_metric]["rho"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step

    # Roll back to the best-SimLex snapshot taken above (unless the peak WAS the
    # final step, in which case the live weights already are the best).
    if cfg.early_stop and best_state is not None and best_step != cfg.steps:
        model.load_state_dict(best_state)
        if verbose:
            print(f"\nEarly stopping: restored best checkpoint from step {best_step} "
                  f"({cfg.early_stop_metric} rho={best_metric:.4f})", flush=True)

    df = pd.DataFrame(history)
    df.attrs["best_step"] = best_step
    return df


def train(cfg: Config, regime: str, shared: Shared | None = None,
          save_tag: str | None = None) -> pd.DataFrame:
    """Build a fresh model and train it. Loads shared data if not provided.

    If `save_tag` is given, the trained weights are written to
    models/bert_{regime}_{save_tag}.pt AFTER any early-stopping rollback - i.e.
    the saved model is the one the reported numbers describe. Needed for any
    downstream use (GLUE fine-tuning, brain/fMRI evaluation), since the training
    history CSVs alone don't let you re-use the model.
    """
    set_seed(cfg.seed)
    print(f"Device: {DEVICE} | torchsort: {HAS_TORCHSORT} | regime: {regime}", flush=True)
    print("Building model ...", flush=True)
    model = BertRSR(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,}", flush=True)
    if shared is None:
        shared = Shared(model.tokenizer, cfg)
    print("\nTraining ...", flush=True)
    df = train_one(cfg, regime, model, shared, verbose=True)

    if save_tag:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = MODELS_DIR / f"bert_{regime}_{save_tag}.pt"
        # Save the underlying BertForMaskedLM state dict (not the BertRSR
        # wrapper) so it loads straight into a stock HF model elsewhere.
        torch.save(model.mlm.state_dict(), out)
        meta = {
            "regime": regime, "combine": cfg.combine, "rsr_lambda": cfg.rsr_lambda,
            "rsr_layer": cfg.rsr_layer, "steps": cfg.steps, "seed": cfg.seed,
            "learning_rate": cfg.learning_rate, "rsr_batch_size": cfg.rsr_batch_size,
            "num_frozen_layers": cfg.num_frozen_layers,
            "projection_dim": cfg.projection_dim, "use_men": cfg.use_men,
            "best_step": int(df.attrs.get("best_step", -1)),
            "model_name": cfg.model_name,
        }
        with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved checkpoint -> {out}", flush=True)
    return df


# ============================================================================
# Sweep: try a few (N, interleave, lr) recipes at reduced steps and report
# which best lifts held-out SimLex over the vanilla baseline.
# ============================================================================
SWEEP_PRESETS = {
    # "combine": which way of mixing MLM+RSR wins (interleave vs weighted lambda).
    "combine": [
        {"name": "interleave 1:1", "combine": "interleave", "rsr_batch_size": 16,
         "interleave_cycle": 2, "mlm_per_cycle": 1, "learning_rate": 5e-5},
        {"name": "weighted L=0.3", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.3, "learning_rate": 5e-5},
        {"name": "weighted L=0.5", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.5, "learning_rate": 5e-5},
        {"name": "weighted L=0.7", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.7, "learning_rate": 5e-5},
        {"name": "weighted L=0.9", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.9, "learning_rate": 5e-5},
    ],
    # "lr": stability sweep around the combine-sweep winner (weighted, low lambda).
    # Everything peaked at step ~100 regardless of lambda, which points at the
    # learning rate / schedule, not the loss design. Lower the LR and eval finely
    # to see if the peak is higher and the overshoot is tameable.
    "lr": [
        {"name": "L0.3 lr5e-5", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.3, "learning_rate": 5e-5},
        {"name": "L0.3 lr3e-5", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.3, "learning_rate": 3e-5},
        {"name": "L0.3 lr1e-5", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.3, "learning_rate": 1e-5},
        {"name": "L0.5 lr2e-5", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.5, "learning_rate": 2e-5},
        {"name": "L0.3 lr5e-6", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.3, "learning_rate": 5e-6},
    ],
    # "layer": WHERE should the RSR loss read from / act? Run the winning
    # weighted recipe but read the target vectors from a different hidden layer
    # each combo (Vulic et al. 2020: middle layers hold the most lexical
    # semantics; a colleague's RoBERTa run found ~layer 5 best, last layer
    # worst). Run --probe-layers first - it's free and narrows which layers
    # are worth sweeping here.
    "layer": [
        {"name": f"read layer {k}", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.7, "learning_rate": 5e-5, "rsr_layer": k}
        for k in (3, 5, 7, 9, 11, 12)
    ],
    # "trainlayer": which SINGLE layer best absorbs the constraint - freeze
    # everything except layer k (RSR also reads from k). Closest to the
    # colleague's unfreeze-one-at-a-time experiment.
    "trainlayer": [
        {"name": f"train only layer {k}", "combine": "weighted", "rsr_batch_size": 16,
         "rsr_lambda": 0.7, "learning_rate": 5e-5, "rsr_layer": k,
         "train_only_layer": k}
        for k in (3, 5, 7, 9, 11, 12)
    ],
}


def run_sweep(base_cfg: Config, kind: str = "combine") -> None:
    from copy import copy

    combos = SWEEP_PRESETS[kind]
    # If --smoke shrank cfg.steps, run a tiny sweep too (wiring check only).
    smoke = base_cfg.steps <= 20
    sweep_steps = 20 if smoke else (1200 if kind == "lr" else 800)
    # The lr sweep evals finely (every 50) to pin down where the real peak is.
    eval_every = 10 if smoke else (50 if kind == "lr" else 100)
    if smoke:
        combos = combos[:2]

    set_seed(base_cfg.seed)
    print(f"Device: {DEVICE} | torchsort: {HAS_TORCHSORT} | SWEEP "
          f"({len(combos)} combos x {sweep_steps} steps, early-stop on)", flush=True)

    # Build one model just to get a tokenizer for Shared; data loaded once.
    print("Building reference model + loading data once ...", flush=True)
    probe = BertRSR(base_cfg)
    shared = Shared(probe.tokenizer, base_cfg)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute output paths up front so we can write incrementally after each
    # combo - if the run dies partway, whatever finished is already on disk.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"bert_sweep_{kind}_{ts}.csv"
    traj_out = RESULTS_DIR / f"bert_sweep_{kind}_{ts}_trajectories.csv"

    rows = []
    traj_frames = []
    for i, combo in enumerate(combos, 1):
        cfg = copy(base_cfg)
        for k, v in combo.items():
            if k != "name":
                setattr(cfg, k, v)
        cfg.steps = sweep_steps
        cfg.eval_every = eval_every
        cfg.early_stop = True
        print(f"\n===== combo {i}/{len(combos)}: {combo['name']} "
              f"(N={cfg.rsr_batch_size}, lr={cfg.learning_rate:.0e}) =====", flush=True)
        set_seed(cfg.seed)
        model = BertRSR(cfg)
        df = train_one(cfg, "rsr", model, shared, verbose=True)
        van = df.iloc[0]["all_rho"]
        best_i = df["all_rho"].idxmax()
        best = df.loc[best_i]
        rows.append({
            "name": combo["name"],
            "vanilla_all": van,
            "best_all": best["all_rho"], "best_step": int(best["step"]),
            "best_neither": best["neither_rho"],
            "delta_all": best["all_rho"] - van,
            "final_all": df.iloc[-1]["all_rho"],
        })
        traj = df[["step", "all_rho", "neither_rho"]].copy()
        traj.insert(0, "name", combo["name"])
        traj_frames.append(traj)
        # Persist after every combo (crash-safe).
        pd.DataFrame(rows).to_csv(out, index=False)
        pd.concat(traj_frames, ignore_index=True).to_csv(traj_out, index=False)
        print(f"  (saved progress: {len(rows)}/{len(combos)} combos -> {out.name})", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    # (out / traj_out already written incrementally above)

    print("\n" + "=" * 78, flush=True)
    print("SWEEP RESULTS  (best held-out SimLex reached during each short run)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'recipe':<18}{'vanilla':>9}{'best':>8}{'@step':>7}"
          f"{'d_all':>8}{'neither':>9}{'final':>8}", flush=True)
    print("-" * 67, flush=True)
    for r in rows:
        print(f"{r['name']:<18}{r['vanilla_all']:>9.3f}{r['best_all']:>8.3f}"
              f"{r['best_step']:>7}{r['delta_all']:>+8.3f}"
              f"{r['best_neither']:>9.3f}{r['final_all']:>8.3f}", flush=True)
    best = max(rows, key=lambda r: r["delta_all"])
    print(f"\nBest recipe by d_all: {best['name']} "
          f"(d_all={best['delta_all']:+.3f} at step {best['best_step']}, "
          f"neither={best['best_neither']:.3f})", flush=True)
    print("\nRead: 'best' is the early-stopped peak; 'final' is end-of-run. A big", flush=True)
    print("best-vs-final gap = overshoot. A lower LR that climbs higher AND keeps a", flush=True)
    print("smaller best-vs-final gap is the stable recipe we want.", flush=True)
    print(f"\nSaved sweep table to: {out}", flush=True)
    print(f"Saved per-step trajectories to: {traj_out}", flush=True)


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


def df_first_last(df: pd.DataFrame, metric: str = "all") -> tuple[dict, dict]:
    """Reconstruct evaluate_simlex-style dicts from a history DataFrame.

    First = vanilla (step 0). Second = the BEST eval row by `{metric}_rho`
    (matches the early-stopped checkpoint), falling back to the final row."""
    def row_to_eval(row) -> dict:
        return {p: {"rho": row[f"{p}_rho"], "n": int(row[f"{p}_n"])} for p in PARTITIONS}
    best_idx = df[f"{metric}_rho"].idxmax()
    return row_to_eval(df.iloc[0]), row_to_eval(df.loc[best_idx])


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["compare", "rsr", "baseline"], default="compare",
                    help="compare (default) = run both vanilla->RSR and vanilla->MLM and "
                         "print the comparison table; rsr / baseline = run just one")
    ap.add_argument("--combine", choices=["interleave", "weighted"], default=None,
                    help="how the RSR regime mixes MLM+RSR: interleave (alternate pure "
                         "batches) or weighted (both per batch, Mark's approach)")
    ap.add_argument("--lambda", dest="rsr_lambda", type=float, default=None,
                    help="weight on RSR term in weighted mode (0..1; Mark used 0.9)")
    ap.add_argument("--steps", type=int, default=None, help="override total optimiser steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rsr-batch-size", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None,
                    help="evaluate SimLex every N steps (default 100, per supervisor)")
    ap.add_argument("--use-men", action="store_true",
                    help="include MEN in the supervision pool (excluded by default: "
                         "it rates relatedness, not strict similarity)")
    ap.add_argument("--frozen-layers", type=int, default=None,
                    help="freeze embeddings + first N encoder layers (default 0 = train "
                         "everything, per supervisor; 11 = Mark's original recipe)")
    ap.add_argument("--proj-dim", type=int, default=None,
                    help="RSR projection head size (default 0 = no head, loss on the raw "
                         "last-layer states, per supervisor; 128 = Mark's original recipe)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="report the final model instead of the best-SimLex checkpoint")
    ap.add_argument("--rsr-layer", type=int, default=None,
                    help="hidden layer the RSR target vectors are read from (0=embeddings, "
                         "1..12=encoder layers; default = last layer). Middle layers often "
                         "carry the most lexical semantics (Vulic et al. 2020)")
    ap.add_argument("--train-only-layer", type=int, default=None,
                    help="freeze everything except this single encoder layer (1..12); the "
                         "MLM head stays trainable. For the layer-localisation experiment")
    ap.add_argument("--save-tag", type=str, default=None,
                    help="save trained weights to models/bert_{regime}_{tag}.pt (+ a .json "
                         "of the config). Saved AFTER early-stopping rollback, so the "
                         "checkpoint matches the reported numbers. Needed for downstream "
                         "/ fMRI evaluation")
    ap.add_argument("--probe-layers", action="store_true",
                    help="NO training: evaluate vanilla SimLex from every hidden layer "
                         "(one forward pass per sentence) and print the per-layer curve. "
                         "Run this first - it's cheap and tells you which --rsr-layer to try")
    ap.add_argument("--sweep", action="store_true",
                    help="run a recipe sweep at reduced steps and report which best lifts "
                         "SimLex (run this first to pick a recipe). See --sweep-kind.")
    ap.add_argument("--sweep-kind", choices=["combine", "lr", "layer", "trainlayer"],
                    default="combine",
                    help="combine = interleave vs weighted lambdas; lr = learning-rate "
                         "stability sweep; layer = read RSR vectors from different hidden "
                         "layers (full backbone trains); trainlayer = freeze all but one "
                         "layer at a time (the unfreeze-one-layer experiment)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run to check everything wires up (CPU-friendly)")
    args = ap.parse_args()

    cfg = Config()
    if args.combine is not None:
        cfg.combine = args.combine
    if args.rsr_lambda is not None:
        cfg.rsr_lambda = args.rsr_lambda
    if args.steps is not None:
        cfg.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.rsr_batch_size is not None:
        cfg.rsr_batch_size = args.rsr_batch_size
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.use_men:
        cfg.use_men = True
    if args.frozen_layers is not None:
        cfg.num_frozen_layers = args.frozen_layers
    if args.proj_dim is not None:
        cfg.projection_dim = args.proj_dim
    if args.no_early_stop:
        cfg.early_stop = False
    if args.rsr_layer is not None:
        cfg.rsr_layer = args.rsr_layer
    if args.train_only_layer is not None:
        cfg.train_only_layer = args.train_only_layer
        # RSR should read from the layer being trained unless told otherwise.
        if args.rsr_layer is None:
            cfg.rsr_layer = args.train_only_layer
    if args.smoke:
        cfg.steps = 20
        cfg.eval_every = 10
        cfg.mlm_batch_size = 4

    if args.probe_layers:
        probe_layers(cfg)
        print("\nDone.", flush=True)
        return

    if args.sweep:
        run_sweep(cfg, kind=args.sweep_kind)
        print("\nDone.", flush=True)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.regime == "compare":
        # Each regime builds its OWN fresh BERT, so the two runs don't
        # contaminate each other; data is loaded once and shared. Vanilla
        # (step-0) eval is identical for both by construction; take it from RSR.
        from transformers import BertTokenizerFast
        shared = Shared(BertTokenizerFast.from_pretrained(cfg.model_name), cfg)
        mode_desc = (f"weighted lambda={cfg.rsr_lambda}" if cfg.combine == "weighted"
                     else f"interleaved {cfg.mlm_per_cycle}MLM:"
                          f"{cfg.interleave_cycle - cfg.mlm_per_cycle}RSR")
        print(f"\n########## RSR REGIME ({mode_desc}) ##########", flush=True)
        rsr_df = train(cfg, "rsr", shared, save_tag=args.save_tag)
        print("\n########## BASELINE REGIME (continued MLM only) ##########", flush=True)
        base_df = train(cfg, "baseline", shared, save_tag=args.save_tag)

        rsr_df.to_csv(RESULTS_DIR / f"bert_continued_rsr_{ts}.csv", index=False)
        base_df.to_csv(RESULTS_DIR / f"bert_continued_baseline_{ts}.csv", index=False)

        vanilla, rsr_final = df_first_last(rsr_df)
        _, mlm_final = df_first_last(base_df)
        print_table2(vanilla, rsr_final, mlm_final)
        print(f"\nSaved per-step histories to: results/bert_continued_*_{ts}.csv", flush=True)
    else:
        df = train(cfg, args.regime, save_tag=args.save_tag)
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
