"""What a training configuration actually trains on. No model is loaded.

Takes the same supervision and sampler switches as the training scripts and
reports:

  - the supervision: scored pairs per source, vocabulary, SimLex pairs dropped
  - the SimLex split: Both / One / Neither by part of speech, and how many Both
    pairs are themselves supervision pairs (and from which source)
  - the RSR signal: simulated batches, scored pairs per batch by source
  - sentence coverage of the supervised words, and of inflected forms

    python experiments/rsr_pretraining/audit_supervision.py
    python experiments/rsr_pretraining/audit_supervision.py --exclude-simlex-pairs \\
        --inflect-verbs --things-pairs 10000 --sampler pairs

The batch simulation skips the step where each word is located in a sentence,
which in training fails for a handful of words, so its numbers are close to but
not exactly what a run sees. Training logs the real per-batch counts.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsr_core import paths
from rsr_core.data import has_sentences, load_simlex, load_simlex_pos, partition, set_seed
from rsr_core.cli import add_supervision_args, apply_args
from rsr_core.sampling import WordPicker, rsr_targets
from rsr_core.supervision import SupervisionOptions, batch_composition, build_supervision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_supervision_args(ap)
    ap.add_argument("--rsr-context", choices=["sentences", "template"], default="sentences")
    ap.add_argument("--rsr-batch-size", type=int, default=24)
    ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-missing", default=None,
                    help="write supervised words that have no harvested sentences to this "
                         "file, one per line, for harvest_sentences_strict.py --targets-file")
    args = ap.parse_args()

    class Cfg:
        use_men = exclude_simlex = inflect_verbs = False
        things_pairs = simverb_share = None
        sampler = "words"
    cfg = Cfg()
    apply_args(cfg, args)

    sup = build_supervision(SupervisionOptions.from_config(cfg))
    print(f"Sentence pool: {paths.sentence_pool()}")
    print(f"Supervision: {sup.summary()}")

    # ---- the SimLex split ------------------------------------------------
    simlex = load_simlex()
    cats = partition(simlex, sup.vocab, lookup=sup.lookup, pos=load_simlex_pos())
    print("\nSimLex-999 split (all 999 pairs, before the sentence-coverage filter)")
    print(f"  {'':<10}{'N':>6}{'V':>6}{'A':>6}{'total':>8}{'pair scored':>14}")
    for p in ("both", "one", "neither"):
        n = {t: len(cats.get(f"{p}_{t}", [])) for t in "NVA"}
        scored = len(cats.get(f"{p}_trained", [])) if p == "both" else 0
        print(f"  {p:<10}{n['N']:>6}{n['V']:>6}{n['A']:>6}{len(cats[p]):>8}{scored:>14}")
    by_src = Counter(sup.source[(a, b)] for a, b, _ in cats.get("both_trained", []))
    if by_src:
        print("  Both pairs that are supervision pairs, by source: "
              + ", ".join(f"{k} {v}" for k, v in sorted(by_src.items())))
    else:
        print("  No SimLex pair is a supervision pair.")

    # ---- coverage --------------------------------------------------------
    targets = rsr_targets(sup, args.rsr_context)
    missing = sorted(w for w in sup.vocab if not has_sentences(w))
    print(f"\nRSR targets: {len(targets):,} of {len(sup.vocab):,} supervised words "
          f"({len(missing):,} have no harvested sentences)")
    infl_words = {w for (a, b), s in sup.source.items() if s == "simverb_infl" for w in (a, b)}
    real_words = {w for (a, b), s in sup.source.items() if s != "simverb_infl" for w in (a, b)}
    new_forms = infl_words - real_words
    if new_forms:
        have = sum(has_sentences(w) for w in new_forms)
        print(f"  inflected forms new to the vocabulary: {len(new_forms):,}, "
              f"{have:,} with sentences")
        usable = set(targets)
        n_infl = sum(1 for (a, b), s in sup.source.items()
                     if s == "simverb_infl" and a < b and a in usable and b in usable)
        print(f"  inflected pairs usable in this context: {n_infl:,} of "
              f"{sup.counts()['simverb_infl']:,}")
    if args.write_missing:
        with open(args.write_missing, "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")
        print(f"  wrote {len(missing):,} words -> {args.write_missing}")

    # ---- simulated batches -----------------------------------------------
    set_seed(args.seed)
    picker = WordPicker(sup, targets, args.rsr_batch_size, cfg.sampler, cfg.simverb_share)
    kept, totals, verb_words = 0, Counter(), 0
    verbs = {w for (a, b), s in sup.source.items() if s.startswith("simverb") for w in (a, b)}
    for _ in range(args.batches):
        words = picker.candidates()[:args.rsr_batch_size]
        if len(words) < args.rsr_batch_size:
            continue
        comp = batch_composition(words, sup.source)
        if sum(comp.values()) < args.min_pairs:
            continue
        kept += 1
        totals.update(comp)
        verb_words += sum(w in verbs for w in words)
    print(f"\nSimulated RSR batches: sampler={cfg.sampler}"
          + (f", simverb share {cfg.simverb_share}" if cfg.simverb_share is not None else "")
          + f", {args.rsr_batch_size} words, context={args.rsr_context}")
    print(f"  kept {kept}/{args.batches} (at least {args.min_pairs} scored pairs)")
    if kept:
        total = sum(totals.values())
        print(f"  scored pairs per batch: {total / kept:.1f}")
        for src, n in sorted(totals.items()):
            print(f"    {src:<14}{n / kept:>8.2f}  ({100 * n / total:.1f}%)")
        print(f"  SimVerb words per batch: {verb_words / kept:.1f} of {args.rsr_batch_size}")


if __name__ == "__main__":
    main()
