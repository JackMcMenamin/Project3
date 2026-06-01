"""
RSR training, SimLex evaluation, and the single-seed driver shared by the
word-level transformer experiments (BERT, GPT-2).

The evaluation partitions SimLex-999 by how many of each pair's words appear
in the RSR supervision vocabulary (both / one / neither in RSR) — this is the
generalisation analysis the paper reports.
"""
from __future__ import annotations

import gc
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .losses import soft_spearman
from .models import DEVICE
from .seeds import set_seed


def train_rsr(model, all_pairs, n_epochs, sample_size, lr, batch_size=64, log_every=50):
    """Optimise the soft-Spearman RSR loss over the supervision pairs."""
    tokenizer = model.tokenizer
    prep = model._prepare  # so "cat" tokenises the same way at train time
    valid_pairs = [
        (w1, w2, s)
        for w1, w2, s in all_pairs
        if tokenizer.tokenize(prep(w1)) and tokenizer.tokenize(prep(w2))
    ]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    for epoch in range(n_epochs):
        model.train()
        sample = (
            random.sample(valid_pairs, sample_size)
            if len(valid_pairs) > sample_size
            else valid_pairs
        )

        unique_words = list({p[0] for p in sample} | {p[1] for p in sample})
        word_to_emb = model.get_batch_embeddings(unique_words, batch_size=batch_size)

        model_sims, human_sims = [], []
        for w1, w2, score in sample:
            if w1 in word_to_emb and w2 in word_to_emb:
                cos = F.cosine_similarity(
                    word_to_emb[w1].unsqueeze(0), word_to_emb[w2].unsqueeze(0)
                )
                model_sims.append(cos)
                human_sims.append(score)

        if len(model_sims) < 10:
            continue

        rho = soft_spearman(
            torch.cat(model_sims), torch.tensor(human_sims, device=DEVICE)
        )
        loss = 1 - rho

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, rho={rho.item():.4f}")


def evaluate_simlex(model, simlex_pairs, rsr_words, batch_size=64) -> dict:
    """Spearman vs human SimLex scores, overall and per RSR-overlap partition."""
    model.eval()

    categories = defaultdict(list)
    for w1, w2, score in simlex_pairs:
        in1, in2 = w1 in rsr_words, w2 in rsr_words
        if in1 and in2:
            cat = "both_in_rsr"
        elif in1 or in2:
            cat = "one_in_rsr"
        else:
            cat = "neither_in_rsr"
        categories["all"].append((w1, w2, score))
        categories[cat].append((w1, w2, score))

    results: dict[str, dict] = {}
    for cat_name, pairs in categories.items():
        all_words = list({p[0] for p in pairs} | {p[1] for p in pairs})
        with torch.no_grad():
            word_to_emb = model.get_batch_embeddings(all_words, batch_size=batch_size)

        model_scores, human_scores = [], []
        for w1, w2, score in pairs:
            if w1 not in word_to_emb or w2 not in word_to_emb:
                continue
            cos = F.cosine_similarity(
                word_to_emb[w1].unsqueeze(0), word_to_emb[w2].unsqueeze(0)
            ).item()
            model_scores.append(cos)
            human_scores.append(score)

        if len(model_scores) < 2:
            results[cat_name] = {"n": 0, "rho": float("nan")}
            continue
        rho, _ = spearmanr(human_scores, model_scores)
        results[cat_name] = {"n": len(model_scores), "rho": rho}

    return results


def run_single_seed(seed, model_factory, all_pairs, rsr_words, simlex_pairs, *, hp) -> dict:
    """Vanilla-vs-RSR for one seed. `model_factory()` builds a fresh wrapper.

    `hp` is any object exposing RSR_EPOCHS / RSR_SAMPLE_SIZE / RSR_LR /
    PROJECTION_DIM / BATCH_SIZE (e.g. a runner's config module).
    """
    print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
    set_seed(seed)

    model = model_factory()

    print("  Evaluating vanilla...")
    vanilla = evaluate_simlex(model, simlex_pairs, rsr_words, batch_size=hp.BATCH_SIZE)

    print("  Training RSR...")
    train_rsr(
        model, all_pairs,
        n_epochs=hp.RSR_EPOCHS, sample_size=hp.RSR_SAMPLE_SIZE,
        lr=hp.RSR_LR, batch_size=hp.BATCH_SIZE,
    )

    print("  Evaluating RSR...")
    rsr = evaluate_simlex(model, simlex_pairs, rsr_words, batch_size=hp.BATCH_SIZE)

    result = {
        "seed": seed,
        "vanilla_all": vanilla["all"]["rho"],
        "vanilla_both": vanilla["both_in_rsr"]["rho"],
        "vanilla_one": vanilla["one_in_rsr"]["rho"],
        "vanilla_neither": vanilla["neither_in_rsr"]["rho"],
        "rsr_all": rsr["all"]["rho"],
        "rsr_both": rsr["both_in_rsr"]["rho"],
        "rsr_one": rsr["one_in_rsr"]["rho"],
        "rsr_neither": rsr["neither_in_rsr"]["rho"],
    }
    for part in ("all", "both", "one", "neither"):
        result[f"delta_{part}"] = result[f"rsr_{part}"] - result[f"vanilla_{part}"]

    print(f"\n  Seed {seed}: All d={result['delta_all']:+.4f}, "
          f"Neither d={result['delta_neither']:+.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
