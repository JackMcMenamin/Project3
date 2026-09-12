"""Is the unsupervised gain contaminated by morphology?

Our "Neither in RSR" partition contains pairs where neither word appears in the
supervision vocabulary. But a word can be absent from that vocabulary and still
be a morphological variant of something in it - `teacher` is unsupervised while
`teach` is supervised - and a sceptic could argue that some of the apparent
generalisation is really cheap morphological transfer.

This script recomputes the headline contrast on a stricter partition that also
removes any pair where either word shares a stem with a supervised word. The
stemmer is deliberately over-eager, so the strict partition throws away more than
it strictly needs to and the resulting number is a conservative floor.

Usage:
    python experiments/rsr_pretraining/eval_strict_neither.py
"""
import os
import sys
from collections import defaultdict
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bert_rsr_continued_pretraining as B

# The module-level sentence pool is only overridden by the --sentences-dir
# flag inside its main(), which we never call, so it must be set here.
# Without this the words are embedded in the ORIGINAL Simple English
# sentences rather than the filtered full-Wikipedia pool the checkpoints
# were trained on.
B.SENTENCES_DIR = B.ARTIFACTS / "sentences_strict"

TAGS = ["strict_l12", "strict_l12_s2", "strict_l12_s3"]

SUFFIXES = ("iness", "ingly", "ation", "ments", "ities", "ously", "ness",
            "ing", "ers", "est", "ies", "ily", "ity", "ful", "ous", "ive",
            "ed", "es", "er", "ly", "al", "s")


def stem(w: str) -> str:
    """Deliberately generous suffix stripper: we want to over-detect
    morphological relatedness so that the strict partition errs toward
    discarding legitimate pairs rather than keeping contaminated ones."""
    w = w.lower()
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            base = w[: -len(suf)]
            return base[:-1] + "y" if base.endswith("i") else base
    return w


def score(ckpt, words, cats, cfg, label):
    model = B.BertRSR(cfg)
    if ckpt is not None:
        model.mlm.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(B.DEVICE)
    emb = B.embed_words_in_context(model, words, cfg, {}, max_sents=20)
    print(f"    {label:<16} embedded {len(emb)}/{len(words)}", flush=True)

    out = {}
    for part, plist in cats.items():
        ms, hs = [], []
        for w1, w2, human in plist:
            if w1 in emb and w2 in emb:
                ms.append(F.cosine_similarity(emb[w1].unsqueeze(0),
                                              emb[w2].unsqueeze(0)).item())
                hs.append(human)
        rho = spearmanr(hs, ms).correlation if len(ms) > 10 else float("nan")
        out[part] = (float(rho), len(ms))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    cfg = B.Config()
    simlex = B.load_simlex()
    _, supervised = B.load_supervision(use_men=cfg.use_men)
    sup_stems = {stem(w) for w in supervised}

    cats = defaultdict(list)
    for w1, w2, human in simlex:
        in1, in2 = w1 in supervised, w2 in supervised
        if in1 and in2:
            cats["both"].append((w1, w2, human))
        elif in1 or in2:
            cats["one"].append((w1, w2, human))
        else:
            cats["neither"].append((w1, w2, human))
            # stricter: also require no morphological link to the supervision
            if stem(w1) not in sup_stems and stem(w2) not in sup_stems:
                cats["neither_strict"].append((w1, w2, human))

    n_all, n_str = len(cats["neither"]), len(cats["neither_strict"])
    print(f"SimLex pairs: {len(simlex)}   supervised vocabulary: {len(supervised)}")
    print(f"neither: {n_all}   neither_strict: {n_str}   "
          f"discarded {n_all - n_str} ({100 * (n_all - n_str) / n_all:.1f}%)")

    words = sorted({w for p in simlex for w in p[:2]})
    print("\nscoring stock BERT")
    vanilla = score(None, words, cats, cfg, "vanilla")

    per_seed = []
    for tag in TAGS:
        c = B.REPO_ROOT / "models" / f"bert_baseline_{tag}.pt"
        r = B.REPO_ROOT / "models" / f"bert_rsr_{tag}.pt"
        if not (os.path.exists(c) and os.path.exists(r)):
            print(f"\nskipping {tag}: checkpoints not on disk")
            continue
        print(f"\nscoring {tag}")
        per_seed.append((tag, score(c, words, cats, cfg, "control"),
                         score(r, words, cats, cfg, "rsr")))

    if not per_seed:
        print("no checkpoints found")
        return

    print()
    print("=" * 72)
    print("Held-out SimLex, RSR minus its matched control, averaged over seeds")
    print("=" * 72)
    print(f"{'partition':<18}{'n':>6}{'vanilla':>10}{'mean':>10}{'sd':>9}   per-seed")
    for part in ("both", "one", "neither", "neither_strict"):
        ds = [r[part][0] - c[part][0] for _, c, r in per_seed]
        sd = stdev(ds) if len(ds) > 1 else float("nan")
        print(f"{part:<18}{vanilla[part][1]:>6}{vanilla[part][0]:>10.4f}"
              f"{mean(ds):>+10.4f}{sd:>9.4f}   "
              + ", ".join(f"{d:+.4f}" for d in ds))

    a = mean([r["neither"][0] - c["neither"][0] for _, c, r in per_seed])
    b = mean([r["neither_strict"][0] - c["neither_strict"][0] for _, c, r in per_seed])
    print()
    print(f"Removing every morphologically related pair changes the unsupervised")
    print(f"gain from {a:+.4f} to {b:+.4f}  (difference {b - a:+.4f}).")


if __name__ == "__main__":
    main()
