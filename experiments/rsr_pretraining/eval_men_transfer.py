"""Does the RSR effect generalise beyond SimLex?

Everything we report rests on held-out SimLex-999, partitioned by our supervision
vocabulary. That leaves an obvious question: have we found something about RSR,
or something about SimLex?

MEN is the sharpest available test, for two reasons.

1. Coverage. 2,980 of its 3,000 pairs use words we already have harvested
   sentences for, so this is evaluation only - no new training, no new harvest.

2. It rates RELATEDNESS, not similarity. Our supervision (SimVerb + THINGS) is
   similarity, and MEN was deliberately excluded from the supervision pool for
   exactly that reason. Similarity and relatedness pull apart on pairs like
   coffee/cup - strongly related, not similar - so a method that installs
   similarity structure need not help here, and could plausibly hurt.

Runs over every seed whose checkpoints are on disk, and reports the per-seed
deltas as well as their mean, so the transfer result does not rest on one run the
way the first version of it did.

Usage:
    python experiments/rsr_pretraining/eval_men_transfer.py
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

# The pool the checkpoints were trained on. It is also the default; set it
# explicitly so this script does not depend on that.
B.paths.set_sentence_pool(B.ARTIFACTS / "sentences_strict")

MEN_PATH = B.DATA_DIR / "MEN" / "MEN" / "MEN_dataset_natural_form_full"
# Seed 1 was saved without a seed suffix; 2 and 3 carry one. Tags whose
# checkpoints are absent are skipped, so this runs with whatever exists.
TAGS = ["strict_l12", "strict_l12_s2", "strict_l12_s3"]
ORDER = ["all", "both", "one", "neither"]


def load_men() -> list[tuple[str, str, float]]:
    """MEN natural-form pairs. Scores are 0-50; Spearman is scale-free so they
    are left as they are."""
    pairs = []
    with open(MEN_PATH, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                w1 = parts[0].rsplit("-", 1)[0].lower()
                w2 = parts[1].rsplit("-", 1)[0].lower()
                pairs.append((w1, w2, float(parts[2])))
    return pairs


def score(ckpt, words, cats, cfg, label) -> dict[str, tuple[float, int]]:
    """Spearman per partition for one checkpoint (None = stock BERT)."""
    model = B.BertRSR(cfg)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        model.mlm.load_state_dict(state, strict=False)
    model.to(B.DEVICE)

    emb = B.embed_words_in_context(model, words, cfg, {}, max_sents=20)
    print(f"    {label:<22} embedded {len(emb)}/{len(words)} words", flush=True)

    out = {}
    for part, plist in cats.items():
        model_s, human_s = [], []
        for w1, w2, human in plist:
            if w1 in emb and w2 in emb:
                cos = F.cosine_similarity(emb[w1].unsqueeze(0), emb[w2].unsqueeze(0)).item()
                model_s.append(cos)
                human_s.append(human)
        rho = spearmanr(human_s, model_s).correlation if len(model_s) > 10 else float("nan")
        out[part] = (float(rho), len(model_s))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    cfg = B.Config()
    pairs = load_men()
    _, supervised = B.load_supervision(use_men=cfg.use_men)
    print(f"MEN pairs: {len(pairs)}   supervised vocabulary: {len(supervised)}")

    # Same three buckets as SimLex, defined by the same supervision vocabulary,
    # so the generalisation question is asked identically of both benchmarks.
    cats: dict[str, list] = defaultdict(list)
    for w1, w2, human in pairs:
        in1, in2 = w1 in supervised, w2 in supervised
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, human))
        cats[cat].append((w1, w2, human))
    print("partitions: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(cats.items())))

    words = sorted({w for p in pairs for w in p[:2]})

    # stock BERT does not depend on the seed, so embed it once
    print("\nscoring stock BERT")
    vanilla = score(None, words, cats, cfg, "vanilla")

    per_seed = []
    for tag in TAGS:
        ctrl_p = B.REPO_ROOT / "models" / f"bert_baseline_{tag}.pt"
        rsr_p = B.REPO_ROOT / "models" / f"bert_rsr_{tag}.pt"
        if not (os.path.exists(ctrl_p) and os.path.exists(rsr_p)):
            print(f"\nskipping {tag}: checkpoints not on disk")
            continue
        print(f"\nscoring {tag}")
        ctrl = score(ctrl_p, words, cats, cfg, "mlm_control")
        rsr = score(rsr_p, words, cats, cfg, "rsr")
        per_seed.append((tag, ctrl, rsr))

    if not per_seed:
        print("\nno checkpoints found - nothing to report")
        return

    print()
    print("=" * 78)
    print("MEN (relatedness), Spearman rho - partitioned by OUR supervision vocabulary")
    print("=" * 78)
    print(f"{'partition':<12}{'n':>7}{'vanilla':>10}" +
          "".join(f"{t.replace('strict_l12', 'seed'):>22}" for t, _, _ in per_seed))
    for part in ORDER:
        row = f"{part:<12}{vanilla[part][1]:>7}{vanilla[part][0]:>10.4f}"
        for _, ctrl, rsr in per_seed:
            row += f"   ctl {ctrl[part][0]:.4f} rsr {rsr[part][0]:.4f}"
        print(row)

    print()
    print("RSR minus its matched control, per seed:")
    print(f"{'partition':<12}{'mean':>9}{'sd':>9}   per-seed")
    for part in ORDER:
        ds = [rsr[part][0] - ctrl[part][0] for _, ctrl, rsr in per_seed]
        sd = stdev(ds) if len(ds) > 1 else float("nan")
        print(f"  {part:<10}{mean(ds):>+9.4f}{sd:>9.4f}   "
              + ", ".join(f"{d:+.4f}" for d in ds))
    print()
    print("For reference, the same contrast on SimLex was +0.070 on Neither.")


if __name__ == "__main__":
    main()
