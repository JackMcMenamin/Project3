"""Similarity datasets, the supervision partition, and harvested sentences."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from . import paths

Pair = tuple[str, str, float]

# The null context from the original rsr_all arm. Every word gets the same
# frame, so its vector is identical on every batch.
RSR_TEMPLATE = "The word is {word}."

PARTITIONS = ("all", "both", "one", "neither")
# Finer splits, scored alongside the four above when the inputs allow:
# Both split by whether the pair itself is a supervision pair, and each
# exposure partition split by SimLex part of speech.
BREAKDOWNS = ("both_trained", "both_unseen",
              "both_N", "both_V", "one_N", "one_V", "neither_N", "neither_V", "neither_A")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _minmax(pairs: list[Pair]) -> list[Pair]:
    if not pairs:
        return pairs
    scores = [p[2] for p in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return pairs
    return [(a, b, (s - lo) / (hi - lo)) for a, b, s in pairs]


def load_men() -> list[Pair]:
    pairs: list[Pair] = []
    if not paths.MEN_PATH.exists():
        return pairs
    with open(paths.MEN_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                w1, w2 = parts[0].split("-")[0], parts[1].split("-")[0]
                pairs.append((w1, w2, float(parts[2])))
    return pairs


def load_simverb() -> list[Pair]:
    pairs: list[Pair] = []
    if not paths.SIMVERB_PATH.exists():
        return pairs
    with open(paths.SIMVERB_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                pairs.append((parts[0], parts[1], float(parts[3])))
    return pairs


def load_things() -> list[Pair]:
    """Every pair of the 1,854 THINGS concepts, scored by SPoSE similarity.
    These scores come from an embedding fitted to odd-one-out judgements, not
    from direct pair ratings."""
    if not paths.THINGS_WORDS_PATH.exists() or not paths.THINGS_SIM_PATH.exists():
        return []
    with open(paths.THINGS_WORDS_PATH, encoding="utf-8") as f:
        words = [ln.strip().replace(" ", "_") for ln in f]
    sim = sio.loadmat(str(paths.THINGS_SIM_PATH))["spose_sim"]
    n = len(words)
    return [(words[i], words[j], float(sim[i, j]))
            for i in range(n) for j in range(i + 1, n)]


def normalise_word(w: str) -> str:
    """Lowercase, with multi-word expressions space-separated (THINGS uses _)."""
    return w.replace("_", " ").lower().strip()


def load_supervision(use_men: bool = False) -> tuple[dict[tuple[str, str], float], set[str]]:
    """SimVerb-3500 and THINGS, each min-max scaled to [0, 1] and pooled: the
    supervision of the reported runs.

    Returns a {(w1, w2): score} lookup holding both orderings, and the set of
    supervised words. MEN is left out unless asked for: it rates relatedness
    rather than similarity, and it is the held-out transfer test. For the
    switchable version see rsr_core.supervision.build_supervision.
    """
    from .supervision import SupervisionOptions, build_supervision
    sup = build_supervision(SupervisionOptions(use_men=use_men))
    return sup.lookup, sup.vocab


def load_simlex() -> list[Pair]:
    pairs: list[Pair] = []
    with open(paths.SIMLEX_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            pairs.append((parts[0].lower(), parts[1].lower(), float(parts[3])))
    return pairs


def load_simlex_pos() -> dict[tuple[str, str], str]:
    """SimLex part of speech (N, V or A) per pair, both orderings."""
    out = {}
    with open(paths.SIMLEX_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            a, b = parts[0].lower(), parts[1].lower()
            out[(a, b)] = out[(b, a)] = parts[2]
    return out


def exposure(w1: str, w2: str, vocab: set[str]) -> str:
    """How many of a pair's two words appear anywhere in the supervision.

    Neither and One pairs cannot have been trained on as pairs. Both pairs
    usually were: SimVerb contains the SimLex verb pairs, and THINGS scores
    every pair of its words, so most Both pairs are supervision pairs too.
    """
    return ("neither", "one", "both")[(w1 in vocab) + (w2 in vocab)]


def partition(pairs: list[Pair], vocab: set[str], lookup: dict | None = None,
              pos: dict | None = None) -> dict[str, list[Pair]]:
    """Split pairs into all / both / one / neither by supervision exposure.

    With `lookup` (the supervision scores), Both is also split into
    both_trained (the pair itself is scored in the supervision) and
    both_unseen. With `pos`, each exposure partition is also split by part of
    speech, as both_N, one_V and so on. Empty splits are left out.
    """
    cats: dict[str, list[Pair]] = defaultdict(list)
    for w1, w2, s in pairs:
        p = (w1, w2, s)
        cats["all"].append(p)
        e = exposure(w1, w2, vocab)
        cats[e].append(p)
        if lookup is not None and e == "both":
            cats["both_trained" if (w1, w2) in lookup else "both_unseen"].append(p)
        if pos is not None and (w1, w2) in pos:
            cats[f"{e}_{pos[(w1, w2)]}"].append(p)
    return cats


def slugify(word: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", word.lower())


def sentence_file(word: str) -> Path:
    return paths.sentence_pool() / f"{slugify(word)}.jsonl"


def has_sentences(word: str) -> bool:
    return sentence_file(word).exists()


_split = {"on": False}


def set_sentence_split(on: bool) -> None:
    """With the split on, part="train" returns the first half of a word's
    sentences and part="eval" the second half, so training never sees an
    evaluation sentence. Off (the reported runs), every part is every sentence."""
    _split["on"] = bool(on)


def sentence_split_on() -> bool:
    return _split["on"]


def load_sentences_for(word: str, part: str = "all") -> list[str]:
    if part not in ("all", "train", "eval"):
        raise ValueError(part)
    out = _read_sentences(word)
    if _split["on"] and part != "all":
        mid = len(out) // 2
        out = out[:mid] if part == "train" else out[mid:]
    return out


def _read_sentences(word: str) -> list[str]:
    path = sentence_file(word)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["sentence"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def find_subword_span(sent_ids: list[int], target_ids: list[int]) -> tuple[int, int] | None:
    """First contiguous occurrence of target_ids in sent_ids, as (start, end)."""
    n, m = len(sent_ids), len(target_ids)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if sent_ids[i:i + m] == target_ids:
            return i, i + m
    return None
