"""RSR on GPT-2: the same recipe, applied to a causal model.

Everything in the project so far is one architecture, which is the largest
remaining limitation. This asks whether the central result -- RSR improves the
pairs it was never supervised on, and not the ones it was -- survives a model
that is autoregressive rather than masked.

WHAT IS HELD CONSTANT, so the numbers are comparable to the BERT ones
-------------------------------------------------------------------
Deliberately imported from the BERT script rather than reimplemented, so there
is no chance of them drifting apart:

  * supervision      SimVerb-3500 + THINGS, 2,652 words, MEN excluded
  * evaluation       held-out SimLex-999, split Both / One / Neither by how much
                     of each pair the supervision touched
  * sentence pool    artifacts/sentences_strict (pass --sentences-dir)
  * the loss itself  B._pair_loss: cosine for every scored pair in the batch,
                     then 1 - soft_spearman against the human ordering
  * pooling          mean over the target word's sub-word tokens, matching BERT
  * hyperparameters  lambda 0.7, 24 words per RSR batch, lr 5e-5, 2000 steps,
                     eval every 100, min 8 scored pairs per batch

WHAT NECESSARILY CHANGES
------------------------
1. The anchor objective. BERT is anchored by masked language modelling; GPT-2
   has no such objective, so the anchor is its own pretraining loss, causal
   language modelling. The recipe is unchanged in form:
       loss = (1 - lambda) * CLM + lambda * RSR
   Substituting the model's own pretraining objective is the minimal defensible
   change: in both cases the anchor is "keep being the language model you were".

2. Attention is causal. A BERT vector for a word sees the whole sentence; a
   GPT-2 vector for a word sees only the tokens up to and including it. So the
   representation is built from left context alone, and a word early in a
   sentence is far less contextualised than a late one. Nothing here corrects
   for that. It is a real difference between the architectures rather than a
   flaw in the setup, and it is one candidate explanation if the result does not
   transfer.

3. Tokenisation. GPT-2 uses byte-level BPE where a leading space is part of the
   token, so " cat" and "cat" are different tokens. Span finding tries the
   space-prefixed form first (the mid-sentence case) and falls back.

4. Casing. bert-base-uncased lowercases everything; GPT-2 is cased. Capitalised
   variants are tried when locating a word, but this remains a genuine
   difference between the two settings and should be stated as such rather than
   waved away.

READ THIS BEFORE CHOOSING A LAYER
---------------------------------
GPT-2's final layer is specialised for next-token prediction and is unusually
anisotropic (Ethayarajh, 2019). That is not a
theoretical worry here. The full probe, all 999 SimLex pairs, 20 sentences per
word (results/gpt2_layer_probe_20260908_005631.csv):

    layer      all   neither    mean cos    sd cos
        0    0.412     0.405       0.721     0.063
        6    0.529     0.540       0.749     0.083   <- best
        7    0.526     0.543       0.760     0.083
       11    0.412     0.453       0.897     0.041
       12    0.161     0.180       0.994     0.006   <- the default

By layer 12 every cosine sits at 0.994 +/- 0.006: the space has collapsed to a
needle and can barely express an ordering, and all-pairs falls to 0.161 against
0.529 six layers below. For comparison, BERT's layer 12 reads 0.480 on the same
pairs, so BERT's final layer is a serviceable lexical read-out and GPT-2's is
not. "Read from the last layer" therefore matches the layer NUMBER across the
two models while mismatching what that layer does.

The default here is still the last layer, deliberately, so nothing is silently
changed relative to the BERT recipe. Pass --rsr-layer explicitly. Note that the
BERT experiments established that the best-probing layer is NOT the best layer
to attach RSR to (layer 7 read better than 12 but gained less, +0.027 against
+0.053), so layer 6 should not simply be assumed correct either. Run both.

Usage:
    # which layer of untouched GPT-2 is already best at SimLex?
    python experiments/contextual_finetune/gpt2_rsr_continued_pretraining.py \
        --probe-layers --sentences-dir artifacts/sentences_strict

    # the main comparison: matched CLM control against RSR
    python experiments/contextual_finetune/gpt2_rsr_continued_pretraining.py \
        --regime compare --lambda 0.7 --seed 1 \
        --sentences-dir artifacts/sentences_strict
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bert_rsr_continued_pretraining as B

DEVICE = B.DEVICE


# ============================================================================
# Config. Mirrors B.Config field for field where the two overlap, so the two
# recipes can be diffed by eye and any divergence is visible.
# ============================================================================
@dataclass
class Config:
    model_name: str = "gpt2"        # 12 layers, 768 hidden: matches bert-base
    rsr_lambda: float = 0.7
    rsr_batch_size: int = 24        # target words per RSR batch
    clm_batch_size: int = 16        # sentences per anchor batch (B: mlm_batch_size)
    learning_rate: float = 5e-5
    soft_rank_strength: float = 1.0
    min_pairs_per_rsr_batch: int = 8
    max_seq_len: int = 128
    steps: int = 2000
    eval_every: int = 100
    seed: int = 1
    projection_dim: int = 0         # 0 = loss acts on the raw 768-d states
    # Hidden layer the RSR target vectors are read from. None = last layer, to
    # match BERT's default. hidden_states is a 13-tuple: [0] embeddings,
    # [1..12] each transformer block.
    rsr_layer: int | None = None
    # Pooling over the word's sub-word tokens.
    #   "mean" - matches BERT exactly, the default so the comparison is clean
    #   "last" - the final sub-word token, which under causal attention is the
    #            only position that has seen the whole word. Kept as a
    #            robustness check rather than the headline.
    pool: str = "mean"
    early_stop: bool = True
    early_stop_metric: str = "all"
    use_men: bool = False


# ============================================================================
# Span finding for byte-level BPE
# ============================================================================
def gpt2_word_variants(word: str) -> list[str]:
    """Surface forms to try when locating a word in a GPT-2 tokenised sentence.

    Space-prefixed first: almost every occurrence is mid-sentence, and in
    byte-level BPE the leading space belongs to the token. The capitalised forms
    catch sentence-initial occurrences, which bert-base-uncased never had to
    worry about.
    """
    return [" " + word, word, " " + word.capitalize(), word.capitalize()]


def find_word_span(tokenizer, sent_ids: list[int], word: str) -> tuple[int, int] | None:
    """First occurrence of `word` in an already-tokenised sentence, or None."""
    for variant in gpt2_word_variants(word):
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        span = B.find_subword_span(sent_ids, ids)
        if span is not None:
            return span
    return None


# ============================================================================
# Model
# ============================================================================
class GPT2RSR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = GPT2TokenizerFast.from_pretrained(cfg.model_name)
        # GPT-2 ships without a pad token. Padding is masked out of the loss and
        # out of attention, so reusing EOS is safe and is the standard fix.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained(cfg.model_name)
        self.lm.config.pad_token_id = self.tokenizer.pad_token_id
        hidden = self.lm.config.n_embd
        self.projection = (
            nn.Linear(hidden, cfg.projection_dim) if cfg.projection_dim > 0
            else nn.Identity()
        )
        self.to(DEVICE)

    # ---- (a) the anchor: causal language modelling ------------------------
    def clm_loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        """GPT2LMHeadModel shifts labels internally, so labels are just the
        input ids with padding set to -100."""
        return self.lm(input_ids=input_ids, attention_mask=attention_mask,
                       labels=labels).loss

    # ---- (b) one contextual vector per target word ------------------------
    def target_vectors(self, input_ids, attention_mask, spans) -> torch.Tensor:
        """N sentences -> N word vectors, pooled over each target word's
        sub-word tokens. Same contract as BertRSR.target_vectors so the shared
        loss code can be used unchanged.

        Note the asymmetry against BERT: under causal attention the vector at a
        given position has seen only the tokens before it, so these vectors
        encode left context only.
        """
        if self.cfg.rsr_layer is None:
            hidden = self.lm.transformer(input_ids=input_ids,
                                         attention_mask=attention_mask).last_hidden_state
        else:
            out = self.lm.transformer(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True)
            hidden = out.hidden_states[self.cfg.rsr_layer]

        vecs = []
        for j, (start, end) in enumerate(spans):
            if self.cfg.pool == "last":
                vecs.append(hidden[j, end - 1])
            else:
                vecs.append(hidden[j, start:end].mean(dim=0))
        return self.projection(torch.stack(vecs))


# ============================================================================
# Anchor batches: plain causal LM over the same sentence pool
# ============================================================================
class CLMBatcher:
    """The counterpart of B.MLMBatcher. No masking is needed: the objective is
    already next-token prediction, so the only bookkeeping is excluding padding
    from the loss."""

    def __init__(self, tokenizer, cfg: Config, sentences: list[str]):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sentences = sentences

    def sample_batch(self):
        sents = random.sample(self.sentences,
                              k=min(len(self.sentences), self.cfg.clm_batch_size))
        enc = self.tokenizer(sents, truncation=True, max_length=self.cfg.max_seq_len,
                             padding=True, return_tensors="pt")
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        labels[enc["attention_mask"] == 0] = -100      # never predict padding
        return (input_ids.to(DEVICE), enc["attention_mask"].to(DEVICE), labels.to(DEVICE))


# ============================================================================
# RSR batches: N target words, each in one of its real sentences
# ============================================================================
class RSRSampler:
    """Mirrors B.RSRBatchSampler: sample N supervised words, put each in one of
    its harvested sentences, and keep the batch only if enough of its pairs have
    human ratings."""

    def __init__(self, tokenizer, cfg: Config, targets: list[str], sim_lookup: dict):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.targets = targets
        self.sim_lookup = sim_lookup
        self._cache: dict[str, list[str]] = {}

    def _sentences(self, word: str) -> list[str]:
        if word not in self._cache:
            self._cache[word] = B.load_sentences_for(word)
        return self._cache[word]

    def sample_batch(self):
        N = self.cfg.rsr_batch_size
        if len(self.targets) < N:
            return None
        words, sents, spans = [], [], []
        pool = random.sample(self.targets, k=min(len(self.targets), N * 6))
        for word in pool:
            if len(words) == N:
                break
            cand = self._sentences(word)
            if not cand:
                continue
            cand = list(cand)
            random.shuffle(cand)
            for sent in cand[:8]:
                ids = self.tokenizer(sent, truncation=True,
                                     max_length=self.cfg.max_seq_len)["input_ids"]
                span = find_word_span(self.tokenizer, ids, word)
                if span is not None:
                    words.append(word)
                    sents.append(sent)
                    spans.append(span)
                    break
        if len(words) < N:
            return None
        scored = sum(1 for a, b in combinations(words, 2) if (a, b) in self.sim_lookup)
        if scored < self.cfg.min_pairs_per_rsr_batch:
            return None
        enc = self.tokenizer(sents, truncation=True, max_length=self.cfg.max_seq_len,
                             padding=True, return_tensors="pt")
        return enc, spans, words


def rsr_loss_for_batch(model: GPT2RSR, batch, spans, words, sim_lookup, cfg):
    vecs = model.target_vectors(batch["input_ids"].to(DEVICE),
                                batch["attention_mask"].to(DEVICE), spans)
    # B._pair_loss is the same soft-Spearman objective the BERT runs use. Reused
    # rather than reimplemented so the two architectures cannot silently end up
    # optimising slightly different things.
    return B._pair_loss(vecs, words, sim_lookup, cfg)


# ============================================================================
# Evaluation: identical partitioning to BERT, GPT-2 embedding
# ============================================================================
def embed_words(model: GPT2RSR, words: list[str], cfg: Config,
                cache: dict[str, list[str]], max_sents: int = 20) -> dict[str, torch.Tensor]:
    model.eval()
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for w in words:
            if w not in cache:
                cache[w] = B.load_sentences_for(w)
            vecs = []
            for sent in cache[w][:max_sents]:
                enc = model.tokenizer(sent, truncation=True,
                                      max_length=cfg.max_seq_len, return_tensors="pt")
                span = find_word_span(model.tokenizer,
                                      enc["input_ids"][0].tolist(), w)
                if span is None:
                    continue
                v = model.target_vectors(enc["input_ids"].to(DEVICE),
                                         enc["attention_mask"].to(DEVICE), [span])[0]
                vecs.append(v)
            if vecs:
                out[w] = torch.stack(vecs).mean(dim=0)
    return out


def evaluate_simlex(model: GPT2RSR, simlex, supervised_vocab, cfg, cache) -> dict:
    """Same three-way split as the BERT evaluation: for each held-out SimLex
    pair, how many of its two words appeared anywhere in the supervision.
    'neither' is the generalisation test."""
    cats = defaultdict(list)
    for w1, w2, s in simlex:
        in1, in2 = w1 in supervised_vocab, w2 in supervised_vocab
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, s))
        cats[cat].append((w1, w2, s))

    results = {}
    for name, pairs in cats.items():
        words = list({w for p in pairs for w in p[:2]})
        emb = embed_words(model, words, cfg, cache)
        model_s, human_s = [], []
        for w1, w2, s in pairs:
            if w1 in emb and w2 in emb:
                model_s.append(F.cosine_similarity(emb[w1].unsqueeze(0),
                                                   emb[w2].unsqueeze(0)).item())
                human_s.append(s)
        if len(model_s) < 2:
            results[name] = {"n": len(model_s), "rho": float("nan")}
        else:
            rho, _ = spearmanr(human_s, model_s)
            results[name] = {"n": len(model_s), "rho": float(rho)}
    return results


def partition(pairs, supervised_vocab):
    """Split any pair list by how much of each pair the supervision touched."""
    cats = defaultdict(list)
    for w1, w2, sc in pairs:
        in1, in2 = w1 in supervised_vocab, w2 in supervised_vocab
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, sc))
        cats[cat].append((w1, w2, sc))
    return cats


def score_pairs(model, cats, cfg, cache) -> dict:
    """Spearman per partition, embedding the union once."""
    words = list({w for p in cats["all"] for w in p[:2]})
    emb = embed_words(model, words, cfg, cache)
    out = {}
    for name, pairs in cats.items():
        ms, hs = [], []
        for w1, w2, h in pairs:
            if w1 in emb and w2 in emb:
                ms.append(F.cosine_similarity(emb[w1].unsqueeze(0),
                                              emb[w2].unsqueeze(0)).item())
                hs.append(h)
        out[name] = {"n": len(ms),
                     "rho": float(spearmanr(hs, ms).correlation) if len(ms) > 10
                            else float("nan")}
    return out


def evaluate_men(model, supervised_vocab, cfg, cache) -> dict:
    """MEN as the second dataset. Never used in training or model selection, and
    it rates relatedness rather than similarity, so it is a genuine transfer
    test rather than a second look at the same thing. Same partition scheme as
    SimLex, so the generalisation signature is directly comparable.

    Run once at the end rather than per step: it is another full embedding pass.
    """
    from eval_men_transfer import load_men
    return score_pairs(model, partition(load_men(), supervised_vocab), cfg, cache)


# ============================================================================
# Layer probe: which layer of untouched GPT-2 is already best at SimLex?
# ============================================================================
def probe_layers(cfg: Config, max_sents: int = 20) -> pd.DataFrame:
    """The GPT-2 counterpart of B.probe_layers. Worth running before choosing a
    read-out: GPT-2's last layer is heavily specialised for next-token
    prediction, so inheriting BERT's layer 12 without checking would be an
    assumption rather than a decision."""
    print("Layer probe: untouched GPT-2, no training, "
          f"{max_sents} sentences/word", flush=True)
    simlex = B.load_simlex()
    _, supervised = B.load_supervision(use_men=cfg.use_men)
    model = GPT2RSR(cfg)
    model.eval()
    NL = model.lm.config.n_layer

    # One forward per sentence with output_hidden_states gives every layer at
    # once, so the whole 13-layer curve costs about one evaluation rather than
    # thirteen. Same trick as B.probe_layers.
    words = sorted({w for p in simlex for w in p[:2]})
    per_layer: dict[int, dict[str, torch.Tensor]] = {L: {} for L in range(NL + 1)}
    with torch.no_grad():
        for i, w in enumerate(words):
            acc = {L: [] for L in range(NL + 1)}
            for sent in B.load_sentences_for(w)[:max_sents]:
                enc = model.tokenizer(sent, truncation=True,
                                      max_length=cfg.max_seq_len, return_tensors="pt")
                span = find_word_span(model.tokenizer, enc["input_ids"][0].tolist(), w)
                if span is None:
                    continue
                out = model.lm.transformer(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                    output_hidden_states=True)
                st, en = span
                for L in range(NL + 1):
                    h = out.hidden_states[L][0, st:en]
                    acc[L].append(h[-1] if cfg.pool == "last" else h.mean(dim=0))
            for L in range(NL + 1):
                if acc[L]:
                    per_layer[L][w] = torch.stack(acc[L]).mean(dim=0)
            if (i + 1) % 200 == 0:
                print(f"  embedded {i+1}/{len(words)} words", flush=True)

    cats = defaultdict(list)
    for w1, w2, s in simlex:
        in1, in2 = w1 in supervised, w2 in supervised
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, s))
        cats[cat].append((w1, w2, s))

    rows = []
    print(f"\n{'layer':>6}{'all':>9}{'both':>9}{'one':>9}{'neither':>9}"
          f"{'mean cos':>11}{'sd cos':>9}")
    for L in range(NL + 1):
        emb = per_layer[L]
        row = {"layer": L}
        allc = []
        for name, pairs in cats.items():
            ms, hs = [], []
            for w1, w2, h in pairs:
                if w1 in emb and w2 in emb:
                    ms.append(F.cosine_similarity(emb[w1].unsqueeze(0),
                                                  emb[w2].unsqueeze(0)).item())
                    hs.append(h)
            row[f"{name}_rho"] = spearmanr(hs, ms).correlation if len(ms) > 10 else float("nan")
            row[f"{name}_n"] = len(ms)
            if name == "all":
                allc = ms
        # Spread matters as much as the correlation: a layer whose cosines are
        # all crammed near 1.0 cannot express an ordering, whatever RSR does.
        row["mean_cos"] = float(np.mean(allc)) if allc else float("nan")
        row["sd_cos"] = float(np.std(allc)) if allc else float("nan")
        rows.append(row)
        print(f"{L:>6}{row['all_rho']:>9.4f}{row['both_rho']:>9.4f}"
              f"{row['one_rho']:>9.4f}{row['neither_rho']:>9.4f}"
              f"{row['mean_cos']:>11.4f}{row['sd_cos']:>9.4f}", flush=True)
    return pd.DataFrame(rows)


# ============================================================================
# Training
# ============================================================================
def train(cfg: Config, regime: str, shared: dict, verbose: bool = True,
          partial_path=None) -> pd.DataFrame:
    """regime: "rsr" (anchor + RSR) or "baseline" (anchor only, the matched
    control). The control is what every reported delta is measured against, so
    it must differ from the RSR arm in exactly one thing: the RSR term."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = GPT2RSR(cfg)
    clm = CLMBatcher(model.tokenizer, cfg, shared["mlm_sentences"])
    rsr = RSRSampler(model.tokenizer, cfg, shared["targets"], shared["sim_lookup"])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)

    cache: dict[str, list[str]] = {}
    history = []
    res = evaluate_simlex(model, shared["simlex"], shared["supervised"], cfg, cache)
    if verbose:
        print("\nBaseline evaluation (before any training):", flush=True)
        print(f"  [step 0] all={res['all']['rho']:.4f} (n={res['all']['n']})  "
              f"both={res['both']['rho']:.4f}  one={res['one']['rho']:.4f}  "
              f"neither={res['neither']['rho']:.4f}", flush=True)
    history.append({"step": 0, **{f"{k}_{m}": res[k][m] for k in res for m in res[k]}})
    # MEN is scored once before and once after. It is a second full embedding
    # pass, too costly to run per step, and only the endpoints are needed.
    men0 = evaluate_men(model, shared["supervised"], cfg, cache)
    if verbose:
        print("  [step 0 MEN] all=%.4f (n=%d)  neither=%.4f"
              % (men0["all"]["rho"], men0["all"]["n"], men0["neither"]["rho"]),
              flush=True)

    best_metric = res[cfg.early_stop_metric]["rho"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step = 0

    run_clm, run_rsr, n_clm, n_rsr = 0.0, 0.0, 0, 0
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        ids, mask, labels = clm.sample_batch()
        clm_term = model.clm_loss(ids, mask, labels)
        run_clm += clm_term.item(); n_clm += 1

        loss = clm_term
        if regime == "rsr":
            sample = rsr.sample_batch()
            if sample is not None:
                batch, spans, words = sample
                rsr_term = rsr_loss_for_batch(model, batch, spans, words,
                                              shared["sim_lookup"], cfg)
                if rsr_term is not None:
                    # identical form to the BERT recipe
                    loss = (1 - cfg.rsr_lambda) * clm_term + cfg.rsr_lambda * rsr_term
                    run_rsr += rsr_term.item(); n_rsr += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0)
        optimizer.step()

        if step % cfg.eval_every == 0:
            clm_avg, rsr_avg = run_clm / max(n_clm, 1), run_rsr / max(n_rsr, 1)
            res = evaluate_simlex(model, shared["simlex"], shared["supervised"], cfg, cache)
            if verbose:
                print(f"step {step}/{cfg.steps}  clm_loss={clm_avg:.4f}  "
                      f"rsr_loss={rsr_avg:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                print(f"  [step {step}] all={res['all']['rho']:.4f}  "
                      f"both={res['both']['rho']:.4f}  one={res['one']['rho']:.4f}  "
                      f"neither={res['neither']['rho']:.4f}", flush=True)
            history.append({"step": step, "clm_loss": clm_avg, "rsr_loss": rsr_avg,
                            **{f"{k}_{m}": res[k][m] for k in res for m in res[k]}})
            # Flush every eval: a run killed partway leaves usable curves rather
            # than nothing. Two power-offs during the BERT runs taught this.
            if partial_path is not None:
                pd.DataFrame(history).to_csv(partial_path, index=False)
            run_clm, run_rsr, n_clm, n_rsr = 0.0, 0.0, 0, 0

            if cfg.early_stop and res[cfg.early_stop_metric]["rho"] > best_metric:
                best_metric = res[cfg.early_stop_metric]["rho"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step

    if cfg.early_stop and best_step != cfg.steps:
        model.load_state_dict(best_state)
        if verbose:
            print(f"\nEarly stopping: restored step {best_step} "
                  f"({cfg.early_stop_metric} rho={best_metric:.4f})", flush=True)

    df = pd.DataFrame(history)
    df.attrs["best_step"] = best_step
    df.attrs["final"] = evaluate_simlex(model, shared["simlex"], shared["supervised"], cfg, cache)
    df.attrs["men_vanilla"] = men0
    df.attrs["men_final"] = evaluate_men(model, shared["supervised"], cfg, cache)
    if verbose:
        print("  [final MEN] all=%.4f  neither=%.4f"
              % (df.attrs["men_final"]["all"]["rho"],
                 df.attrs["men_final"]["neither"]["rho"]), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def load_shared(cfg: Config) -> dict:
    """All data comes from the BERT module, so both architectures are trained
    and scored on exactly the same inputs."""
    simlex = B.load_simlex()
    sim_lookup, supervised = B.load_supervision(use_men=cfg.use_men)
    targets = sorted(w for w in supervised
                     if (B.SENTENCES_DIR / f"{B.slugify(w)}.jsonl").exists())
    print(f"  supervised vocab: {len(supervised)} | SimLex pairs: {len(simlex)}", flush=True)
    print(f"  usable RSR targets (supervised + have sentences): {len(targets)}", flush=True)

    # The anchor corpus, built exactly as B.Shared builds it: sentences harvested
    # for 400 randomly chosen target words. Same construction, same size, so the
    # two architectures see the same kind of text.
    print("Loading anchor sentence pool ...", flush=True)
    mlm_sentences = []
    for w in random.sample(targets, k=min(400, len(targets))):
        mlm_sentences.extend(B.load_sentences_for(w))
    print(f"  anchor sentences: {len(mlm_sentences)}", flush=True)

    return {"simlex": simlex, "sim_lookup": sim_lookup, "supervised": supervised,
            "targets": targets, "mlm_sentences": mlm_sentences}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["compare", "rsr", "baseline"], default="compare")
    ap.add_argument("--lambda", dest="rsr_lambda", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rsr-layer", type=int, default=None,
                    help="hidden layer the RSR vectors are read from (0=embeddings, "
                         "1..12=blocks; default last, matching BERT). GPT-2's last "
                         "layer is specialised for next-token prediction, so this is "
                         "worth probing rather than assuming")
    ap.add_argument("--pool", choices=["mean", "last"], default=None,
                    help="pool sub-word tokens by mean (default, matches BERT) or take "
                         "the last, which under causal attention is the only position "
                         "that has seen the whole word")
    ap.add_argument("--model-name", default=None, help="default gpt2 (12 layers, 768d)")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--probe-layers", action="store_true")
    ap.add_argument("--sentences-dir", default=None,
                    help="sentence pool. The default is the SIMPLE ENGLISH corpus; "
                         "pass artifacts/sentences_strict to match every current result")
    ap.add_argument("--save-tag", default=None)
    args = ap.parse_args()

    if args.sentences_dir:
        B.SENTENCES_DIR = B.REPO_ROOT / args.sentences_dir
        if not B.SENTENCES_DIR.is_dir():
            raise SystemExit(f"no such sentence pool: {B.SENTENCES_DIR}")
        n = sum(1 for _ in B.SENTENCES_DIR.glob("*.jsonl"))
        print(f"Sentence pool: {B.SENTENCES_DIR}  ({n} target files)", flush=True)

    cfg = Config()
    for flag, attr in [("rsr_lambda", "rsr_lambda"), ("steps", "steps"),
                       ("eval_every", "eval_every"), ("seed", "seed"),
                       ("rsr_layer", "rsr_layer"), ("pool", "pool"),
                       ("model_name", "model_name")]:
        v = getattr(args, flag)
        if v is not None:
            setattr(cfg, attr, v)
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.no_early_stop:
        cfg.early_stop = False

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.save_tag or "gpt2"

    if args.probe_layers:
        df = probe_layers(cfg)
        out = B.RESULTS_DIR / f"gpt2_layer_probe_{stamp}.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved -> {out}")
        return

    shared = load_shared(cfg)
    frames = {}
    for regime in (["baseline", "rsr"] if args.regime == "compare" else [args.regime]):
        print(f"\n########## GPT-2 {regime.upper()} REGIME "
              f"(lambda={cfg.rsr_lambda}) ##########", flush=True)
        frames[regime] = train(
            cfg, regime, shared,
            partial_path=B.RESULTS_DIR / f"partial_gpt2_{regime}_{tag}.csv")
        out = B.RESULTS_DIR / f"gpt2_continued_{regime}_{tag}_{stamp}.csv"
        frames[regime].to_csv(out, index=False)
        print(f"Saved -> {out}", flush=True)

    if len(frames) == 2:
        bf, rf = frames["baseline"], frames["rsr"]
        van_s = rf.iloc[0]          # step-0 row: vanilla GPT-2, before training
        # Two datasets, same partition scheme. SimLex measures similarity and is
        # what the supervision is drawn from the same family as; MEN measures
        # relatedness and was never used in training or model selection, so it is
        # a genuine transfer test rather than a second look at the same thing.
        panels = [
            ("SimLex-999 (held out)",
             {k: van_s[k + "_rho"] for k in ["all", "both", "one", "neither"]},
             bf.attrs["final"], rf.attrs["final"]),
            ("MEN (never used in training or model selection)",
             {k: rf.attrs["men_vanilla"][k]["rho"]
              for k in ["all", "both", "one", "neither"]},
             bf.attrs["men_final"], rf.attrs["men_final"]),
        ]
        for title, van, base, rsr in panels:
            print("\n" + "=" * 76)
            print("GPT-2  |  " + title)
            print("=" * 76)
            print(f"{'Category':<18}{'Vanilla':>10}{'Control':>10}"
                  f"{'RSR':>10}{'RSR-ctl':>10}")
            for k, label in [("all", "All pairs"), ("both", "Both in RSR"),
                             ("one", "One in RSR"), ("neither", "Neither in RSR")]:
                print(f"{label:<18}{van[k]:>10.4f}{base[k]['rho']:>10.4f}"
                      f"{rsr[k]['rho']:>10.4f}{rsr[k]['rho'] - base[k]['rho']:>+10.4f}")
        print("\n  'Neither in RSR' is the generalisation test. BERT gives +0.070")
        print("  on SimLex and +0.062 on MEN; those are the rows to compare with.")


if __name__ == "__main__":
    main()
