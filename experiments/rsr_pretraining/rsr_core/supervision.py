"""The supervision set, with switches for what goes into it.

With every option at its default this is exactly the set the reported runs used
(load_supervision returns its lookup and vocab): all of SimVerb-3500 and every
pair of THINGS, each min-max scaled, nothing filtered.

The options:

exclude_simlex
    Drop every supervision pair that is also a SimLex-999 pair, in either
    order. With inflect_verbs, a base pair that is a SimLex pair takes all its
    inflected copies with it, and an inflected pair that happens to equal a
    SimLex pair is dropped too. Afterwards no SimLex pair is ever a training
    target, so Both means "both words seen, this pair never scored".

inflect_verbs
    Add third person, past and gerund forms of every SimVerb pair with the
    base pair's score (run/jump -> runs/jumps, ran/jumped, running/jumping),
    via lemminflect, as the original RoBERTa pipeline does. Inflected forms
    are new words: in the sentence context they need harvested sentences
    (see experiments/contextual/harvest_sentences_strict.py --targets-file).

things_pairs
    Keep a fixed random sample of this many THINGS pairs instead of all 1.7M.
    The sample uses its own seed (things_seed), so every run and both arms of
    a comparison train on the same THINGS pairs. The original pipeline keeps
    10,000. With word-first sampling a small sample leaves most batches below
    the minimum number of scored pairs, so use it with the pair sampler.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from .data import (_minmax, has_sentences, load_men, load_simlex, load_simverb,
                   load_things, normalise_word)


@dataclass
class SupervisionOptions:
    use_men: bool = False
    exclude_simlex: bool = False
    inflect_verbs: bool = False
    things_pairs: int | None = None
    things_seed: int = 42

    @classmethod
    def from_config(cls, cfg) -> "SupervisionOptions":
        return cls(use_men=cfg.use_men, exclude_simlex=cfg.exclude_simlex,
                   inflect_verbs=cfg.inflect_verbs, things_pairs=cfg.things_pairs)


@dataclass
class Supervision:
    lookup: dict[tuple[str, str], float]     # both orderings
    source: dict[tuple[str, str], str]       # both orderings: simverb, simverb_infl, things, men
    vocab: set[str]
    options: SupervisionOptions
    dropped_simlex: Counter = field(default_factory=Counter)

    def pairs(self, source: str | None = None) -> list[tuple[str, str]]:
        """Each scored pair once (a < b), optionally from one source only."""
        return [(a, b) for (a, b), s in self.source.items()
                if a < b and (source is None or s == source)]

    def counts(self) -> Counter:
        return Counter(s for (a, b), s in self.source.items() if a < b)

    def summary(self) -> str:
        c = self.counts()
        parts = ", ".join(f"{k} {v:,}" for k, v in sorted(c.items()))
        out = f"{sum(c.values()):,} scored pairs ({parts}) over {len(self.vocab):,} words"
        if self.options.exclude_simlex:
            d = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped_simlex.items())) or "none"
            out += f"; SimLex pairs dropped: {d}"
        return out


_VERB_TAGS = (("VBZ", 0), ("VBD", -1), ("VBG", 0))


def _inflect(word: str, tag: str, pick: int) -> str:
    import lemminflect
    if word == "sting" and tag == "VBG":    # lemminflect returns nothing here
        return "stinging"
    forms = lemminflect.getInflection(word, tag=tag)
    return forms[pick] if forms else word


def inflect_pairs(pairs):
    """Base pairs plus their VBZ, VBD and VBG forms at the same score. The
    second item of each output is the base pair it came from."""
    out = []
    for w1, w2, s in pairs:
        out.append(((w1, w2, s), (w1, w2)))
        for tag, pick in _VERB_TAGS:
            i1, i2 = _inflect(w1, tag, pick), _inflect(w2, tag, pick)
            if (i1, i2) != (w1, w2):
                out.append(((i1, i2, s), (w1, w2)))
    return out


def _simlex_keys() -> set[tuple[str, str]]:
    keys = set()
    for w1, w2, _ in load_simlex():
        a, b = normalise_word(w1), normalise_word(w2)
        keys.add((a, b))
        keys.add((b, a))
    return keys


def build_supervision(opts: SupervisionOptions | None = None) -> Supervision:
    opts = opts or SupervisionOptions()
    simlex = _simlex_keys() if opts.exclude_simlex else set()

    # (pair, base pair, source). Order matters: where a pair is in two sources
    # the later one sets the score, as it always has.
    entries = []
    simverb = _minmax(load_simverb())
    if opts.inflect_verbs:
        for p, base in inflect_pairs(simverb):
            entries.append((p, base, "simverb" if p[:2] == base else "simverb_infl"))
    else:
        entries += [(p, p[:2], "simverb") for p in simverb]

    things = _minmax(load_things())
    if opts.things_pairs is not None and opts.things_pairs < len(things):
        things = random.Random(opts.things_seed).sample(things, opts.things_pairs)
    entries += [(p, p[:2], "things") for p in things]

    if opts.use_men:
        entries += [(p, p[:2], "men") for p in _minmax(load_men())]

    lookup, source, vocab, dropped = {}, {}, set(), Counter()
    for (w1, w2, s), (b1, b2), src in entries:
        a, b = normalise_word(w1), normalise_word(w2)
        if simlex and ((a, b) in simlex or
                       (normalise_word(b1), normalise_word(b2)) in simlex):
            dropped[src] += 1
            continue
        if src == "simverb_infl" and (a, b) in source:
            continue        # an inflected copy never overrides a real pair
        lookup[(a, b)] = lookup[(b, a)] = s
        source[(a, b)] = source[(b, a)] = src
        vocab.update((a, b))
    return Supervision(lookup, source, vocab, opts, dropped)


def anchor_words(sup: Supervision) -> list[str]:
    """Words whose sentences make up the anchor (MLM or CLM) corpus: the
    default supervision's words that have sentences, whatever switches `sup`
    was built with, so changing the supervision leaves the anchor text alone."""
    default = SupervisionOptions(use_men=sup.options.use_men)
    vocab = sup.vocab if sup.options == default else build_supervision(default).vocab
    return sorted(w for w in vocab if has_sentences(w))


def batch_composition(words: list[str], source: dict) -> Counter:
    """Scored pairs in a batch of words, counted by source."""
    c = Counter()
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            s = source.get((a, b))
            if s is not None:
                c[s] += 1
    return c
