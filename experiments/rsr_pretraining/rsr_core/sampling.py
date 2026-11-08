"""Which supervised words make up an RSR batch.

The model scripts put each candidate word in a sentence and keep the first N
that can be located; this module only decides the candidates and their order,
so BERT and GPT-2 share it.

"words" (the reported runs)
    N * 6 supervised words drawn uniformly; the batch scores whichever pairs
    among them happen to be rated. THINGS rates every pair of its words and
    SimVerb about 1% of verb pairs, so almost every scored pair is a THINGS
    pair.

"pairs" (the original RoBERTa sampler)
    Draw scored pairs and take both words of each, in order, until there are
    enough words. Every other rated pair among the chosen words is scored too.
    simverb_share fixes the fraction of drawn pairs that come from SimVerb
    (including inflected copies); left as None, pairs are drawn uniformly
    over the whole supervision.
"""
from __future__ import annotations

import random

from .data import has_sentences
from .supervision import Supervision

SAMPLERS = ("words", "pairs")


def rsr_targets(sup: Supervision, rsr_context: str) -> list[str]:
    """Supervised words that can go into an RSR batch: those with harvested
    sentences. In the template context, words that only exist as inflected
    forms are kept too, since the template needs no harvest."""
    extra: set[str] = set()
    if rsr_context == "template":
        real = {w for (a, b), s in sup.source.items() if s != "simverb_infl" for w in (a, b)}
        extra = sup.vocab - real
    return sorted(w for w in sup.vocab if w in extra or has_sentences(w))


class WordPicker:
    def __init__(self, sup: Supervision, targets: list[str], n_words: int,
                 mode: str = "words", simverb_share: float | None = None):
        if mode not in SAMPLERS:
            raise ValueError(f"unknown sampler {mode!r}")
        if simverb_share is not None and mode != "pairs":
            raise ValueError("simverb_share needs the pair sampler")
        self.targets = targets
        self.n_words = n_words
        self.mode = mode
        self.simverb_share = simverb_share
        if mode == "pairs":
            usable = set(targets)
            pairs = [(a, b, s) for (a, b), s in sup.source.items()
                     if a < b and a in usable and b in usable]
            self.all_pairs = [(a, b) for a, b, _ in pairs]
            self.verb_pairs = [(a, b) for a, b, s in pairs if s.startswith("simverb")]
            self.other_pairs = [(a, b) for a, b, s in pairs if not s.startswith("simverb")]
            if simverb_share is not None and not (self.verb_pairs and self.other_pairs):
                raise ValueError("simverb_share needs both SimVerb and other pairs")

    def _draw_pair(self) -> tuple[str, str]:
        if self.simverb_share is None:
            return random.choice(self.all_pairs)
        group = self.verb_pairs if random.random() < self.simverb_share else self.other_pairs
        return random.choice(group)

    def candidates(self) -> list[str]:
        N = self.n_words
        if self.mode == "words":
            return random.sample(self.targets, k=min(len(self.targets), N * 6))
        # Twice the words needed, so a word that cannot be placed in any of its
        # sentences does not leave the batch short.
        out, seen = [], set()
        for _ in range(N * 20):
            if len(out) >= 2 * N:
                break
            for w in self._draw_pair():
                if w not in seen:
                    seen.add(w)
                    out.append(w)
        return out
