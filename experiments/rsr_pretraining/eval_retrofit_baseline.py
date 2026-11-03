"""Retrofitting as a baseline for RSR.

Our Related Work argues that RSR differs from retrofitting because the constraint
is applied to the encoder during training rather than to a fixed table of vectors
afterwards. That is a claim about generalisation, and it deserves a measurement
rather than an assertion.

This runs retrofitting (Faruqui et al., 2015) over the same supervision and
evaluates it on the same partition of held-out SimLex:

    q_i  <-  ( a_i * qhat_i  +  sum_j b_ij * q_j ) / ( a_i + sum_j b_ij )

Two adaptations are needed, and both are chosen to be generous to retrofitting
rather than convenient for us.

First, retrofitting expects a sparse lexicon of related-word links, whereas our
supervision is a dense graded similarity matrix: 17 percent of all pairs score
above 0.5, so a threshold graph would connect almost everything to everything and
collapse the space. We therefore build a k-nearest-neighbour graph instead, which
is the standard construction for a dense matrix, and sweep k.

Second, the graph is built over the whole supervision vocabulary rather than only
the words appearing in SimLex, so retrofitting gets every constraint it could
possibly use. Edge weights are the human similarities themselves, so that
strongly similar pairs pull harder.

The structural point to watch is that retrofitting can only move vectors that
appear in its graph. Words that were never supervised have no edges, so their
vectors cannot change at all, whatever k is chosen.

Usage:
    python experiments/rsr_pretraining/eval_retrofit_baseline.py
"""
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bert_rsr_continued_pretraining as B

# The pool the checkpoints were trained on. It is also the default; set it
# explicitly so this script does not depend on that.
B.paths.set_sentence_pool(B.ARTIFACTS / "sentences_strict")

KS = [3, 5, 10, 20]
ITERS = 10
ORDER = ["all", "both", "one", "neither"]


def partitions(simlex, supervised):
    cats = defaultdict(list)
    for w1, w2, human in simlex:
        in1, in2 = w1 in supervised, w2 in supervised
        cat = "both" if (in1 and in2) else ("one" if (in1 or in2) else "neither")
        cats["all"].append((w1, w2, human))
        cats[cat].append((w1, w2, human))
    return cats


def score_vectors(emb, cats):
    out = {}
    for part, plist in cats.items():
        ms, hs = [], []
        for w1, w2, human in plist:
            if w1 in emb and w2 in emb:
                ms.append(F.cosine_similarity(emb[w1].unsqueeze(0),
                                              emb[w2].unsqueeze(0)).item())
                hs.append(human)
        rho = spearmanr(hs, ms).correlation if len(ms) > 10 else float("nan")
        out[part] = float(rho)
    return out


def mean_cosines(emb, cats):
    """Mean cosine per partition. Spearman only sees ordering, so it cannot show
    whether a method has shifted one region of the space onto a different scale
    from the rest. That shift is what breaks comparisons spanning two regions,
    so it needs its own number."""
    out = {}
    for part, plist in cats.items():
        ms = [F.cosine_similarity(emb[w1].unsqueeze(0), emb[w2].unsqueeze(0)).item()
              for w1, w2, _ in plist if w1 in emb and w2 in emb]
        out[part] = sum(ms) / len(ms) if ms else float("nan")
    return out


def knn_graph(sim_lookup, have_vec, k):
    """Top-k most similar supervised neighbours for each word we have a vector
    for. Symmetrised, so a link kept by either endpoint survives."""
    cand = defaultdict(list)
    for (w1, w2), sc in sim_lookup.items():
        if w1 in have_vec and w2 in have_vec:
            cand[w1].append((sc, w2))
    graph = defaultdict(dict)
    for w, lst in cand.items():
        lst.sort(reverse=True)
        for sc, nb in lst[:k]:
            graph[w][nb] = sc
            graph[nb][w] = sc          # symmetrise
    edges = sum(len(v) for v in graph.values()) // 2
    return graph, edges


def retrofit(emb, graph, iters=ITERS):
    q = {w: v.clone() for w, v in emb.items()}
    for _ in range(iters):
        for w, nbrs in graph.items():
            if not nbrs:
                continue
            num, den = emb[w].clone(), 1.0     # alpha = 1 anchors to the original
            for nb, beta in nbrs.items():
                num = num + beta * q[nb]
                den += beta
            q[w] = num / den
    return q


def main() -> None:
    cfg = B.Config()
    simlex = B.load_simlex()
    sim_lookup, supervised = B.load_supervision(use_men=cfg.use_men)
    cats = partitions(simlex, supervised)

    simlex_words = {w for p in simlex for w in p[:2]}
    # embed the SimLex words plus every supervised word we have sentences for,
    # so the retrofitting graph is the full one rather than a SimLex slice
    have_sents = {w for w in supervised
                  if B.has_sentences(w)}
    words = sorted(simlex_words | have_sents)
    print(f"SimLex pairs {len(simlex)}   supervised vocab {len(supervised)}")
    print("partitions: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(cats.items())))
    print(f"embedding {len(words)} words "
          f"({len(simlex_words)} from SimLex, {len(have_sents)} supervised with sentences)")

    model = B.BertRSR(cfg)
    model.to(B.DEVICE)
    emb = B.embed_words_in_context(model, words, cfg, {}, max_sents=20)
    print(f"  embedded {len(emb)}/{len(words)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base = score_vectors(emb, cats)
    rows = []
    for k in KS:
        graph, edges = knn_graph(sim_lookup, emb, k)
        q = retrofit(emb, graph)
        rows.append((k, len(graph), edges, score_vectors(q, cats), mean_cosines(q, cats)))

    print()
    print("=" * 80)
    print("Retrofitting BERT's word vectors over the same supervision")
    print("=" * 80)
    print(f"{'condition':<18}{'words':>7}{'edges':>8}" + "".join(f"{p:>11}" for p in ORDER))
    print(f"{'stock BERT':<18}{'':>7}{'':>8}" + "".join(f"{base[p]:>11.4f}" for p in ORDER))
    for k, nw, ne, sc, _ in rows:
        print(f"{'retrofit k=' + str(k):<18}{nw:>7}{ne:>8}"
              + "".join(f"{sc[p]:>11.4f}" for p in ORDER))

    print()
    print("change against stock BERT:")
    print(f"{'condition':<18}" + "".join(f"{p:>11}" for p in ORDER))
    for k, _, _, sc, _ in rows:
        print(f"{'retrofit k=' + str(k):<18}"
              + "".join(f"{sc[p] - base[p]:>+11.4f}" for p in ORDER))

    # Where does the all-pairs collapse come from? Not from any partition getting
    # worse on its own. Retrofitting pulls the supervised words towards their
    # neighbours, which compresses that region and raises every cosine inside it,
    # while the unsupervised words stay exactly where they were. Once the two
    # regions sit on different scales, every comparison that spans them is wrong,
    # and all-pairs Spearman is mostly such comparisons.
    base_mc = mean_cosines(emb, cats)
    print()
    print("mean cosine per partition (the scale, not the ordering):")
    print(f"{'condition':<18}" + "".join(f"{p:>11}" for p in ORDER))
    print(f"{'stock BERT':<18}" + "".join(f"{base_mc[p]:>11.4f}" for p in ORDER))
    for k, _, _, _, mc in rows:
        print(f"{'retrofit k=' + str(k):<18}"
              + "".join(f"{mc[p]:>11.4f}" for p in ORDER))
    bk = max(rows, key=lambda r: r[3]["both"] - base["both"])
    print(f"{'shift, k=' + str(bk[0]):<18}"
          + "".join(f"{bk[4][p] - base_mc[p]:>+11.4f}" for p in ORDER))

    best = max(rows, key=lambda r: r[3]["both"] - base["both"])
    k, _, _, sc, _ = best
    print()
    print(f"Best retrofitting result, k={k}: "
          f"{sc['both'] - base['both']:+.4f} on supervised pairs, "
          f"{sc['neither'] - base['neither']:+.4f} on unsupervised.")
    print("RSR on the same evaluation gives -0.009 supervised, +0.070 unsupervised.")


if __name__ == "__main__":
    main()
