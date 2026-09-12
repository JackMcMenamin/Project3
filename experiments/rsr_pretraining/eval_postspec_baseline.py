"""Post-specialisation as a baseline for RSR.

Our retrofitting comparison shows exactly 0.000 on unsupervised pairs, which is
arithmetically forced: those words have no edges in the graph, so nothing moves
them. That limitation is not our discovery. Vulic, Glavas, Mrksic and Korhonen
(NAACL 2018) start from it, and post-specialisation is their answer:

    1. specialise the vectors of words the lexicon covers (here, retrofitting)
    2. learn a function mapping original vectors to specialised ones, fitted
       ONLY on those words
    3. apply that function to every other word

The learned map is what reaches the unseen vocabulary, so this is a far stronger
baseline than retrofitting alone, and it is the comparison a reviewer is most
likely to ask for. It is also the fair test of our Related Work claim: RSR
changes the encoder during training, post-specialisation changes a fixed table
and then learns to extrapolate from it. Whether changing the encoder buys
anything over extrapolating is exactly what this measures.

Design choices, all made in post-specialisation's favour:

  * the mapping is trained on every supervised word we have a vector for, not
    just the SimLex slice, so it sees as much of the specialisation as possible
  * a held-out split of supervised words monitors the fit, so the reported map
    is the one that generalised best rather than the last epoch
  * the retrofitting stage reuses the same k-NN graph and the same best k as the
    retrofitting baseline, so the two rows are directly comparable
  * the map is a two-layer network, matching the spirit of the original, which
    used a deep feed-forward specialisation function

Usage:
    python experiments/rsr_pretraining/eval_postspec_baseline.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bert_rsr_continued_pretraining as B

# see eval_men_transfer.py: without this, words are embedded in the original
# Simple English sentences rather than the strict pool, and nothing is
# comparable to any current result
B.SENTENCES_DIR = B.ARTIFACTS / "sentences_strict"

import eval_retrofit_baseline as R  # noqa: E402  (import after SENTENCES_DIR)

ORDER = ["all", "both", "one", "neither"]
K = 3               # the k that most favoured retrofitting in the baseline
EPOCHS = 300
HIDDEN = 768
LR = 1e-3
VAL_FRAC = 0.1


class SpecMap(nn.Module):
    """The specialisation function: original vector -> specialised vector.

    A residual formulation, so the map starts as the identity and learns the
    displacement. Predicting the vector outright makes the easiest solution
    "return the input", which is exactly the degenerate answer we need to be
    able to rule out.
    """

    def __init__(self, dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)


def train_map(orig: torch.Tensor, spec: torch.Tensor, verbose: bool = True) -> SpecMap:
    """Fit the map on supervised words, holding some out to pick the checkpoint."""
    n = orig.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * VAL_FRAC))
    val, tr = perm[:n_val], perm[n_val:]
    xtr, ytr = orig[tr], spec[tr]
    xva, yva = orig[val], spec[val]

    model = SpecMap(orig.shape[1]).to(B.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state = float("inf"), None
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(xva), yva).item()
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
        if verbose and ep % 50 == 0:
            print(f"    epoch {ep:>4}  train {loss.item():.5f}  held-out {v:.5f}", flush=True)
    model.load_state_dict(best_state)
    if verbose:
        print(f"    best held-out MSE {best:.5f}", flush=True)
    return model


def main() -> None:
    cfg = B.Config()
    simlex = B.load_simlex()
    sim_lookup, supervised = B.load_supervision(use_men=cfg.use_men)
    cats = R.partitions(simlex, supervised)

    simlex_words = {w for p in simlex for w in p[:2]}
    have_sents = {w for w in supervised
                  if (B.SENTENCES_DIR / f"{B.slugify(w)}.jsonl").exists()}
    words = sorted(simlex_words | have_sents)
    print(f"SimLex pairs {len(simlex)}   supervised vocab {len(supervised)}")
    print("partitions: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(cats.items())))

    model = B.BertRSR(cfg)
    model.to(B.DEVICE)
    emb = B.embed_words_in_context(model, words, cfg, {}, max_sents=20)
    print(f"  embedded {len(emb)}/{len(words)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base = R.score_vectors(emb, cats)

    # ---- stage 1: retrofit the words the supervision covers ----------------
    graph, edges = R.knn_graph(sim_lookup, emb, K)
    retro = R.retrofit(emb, graph)
    retro_scores = R.score_vectors(retro, cats)
    print(f"\nstage 1: retrofitting, k={K}, {len(graph)} words, {edges} edges")

    # ---- stage 2: learn original -> specialised, on those words only -------
    fitted = sorted(graph.keys())
    print(f"stage 2: fitting the specialisation map on {len(fitted)} supervised words")
    X = torch.stack([emb[w] for w in fitted]).to(B.DEVICE)
    Y = torch.stack([retro[w] for w in fitted]).to(B.DEVICE)
    spec_map = train_map(X, Y)

    # ---- stage 3: apply it to EVERY word, including the unseen ones --------
    with torch.no_grad():
        allw = sorted(emb.keys())
        mapped = spec_map(torch.stack([emb[w] for w in allw]).to(B.DEVICE))
        post = {w: mapped[i].cpu() for i, w in enumerate(allw)}
    n_unseen = sum(1 for w in allw if w not in graph)
    print(f"stage 3: map applied to {len(allw)} words, "
          f"{n_unseen} of which the specialisation never touched")
    post_scores = R.score_vectors(post, cats)

    # ---- report ------------------------------------------------------------
    print()
    print("=" * 84)
    print("Post-specialisation (Vulic et al., 2018) against retrofitting and RSR")
    print("=" * 84)
    print(f"{'condition':<26}" + "".join(f"{p:>13}" for p in ORDER))
    print(f"{'stock BERT':<26}" + "".join(f"{base[p]:>13.4f}" for p in ORDER))
    print(f"{'retrofitting (stage 1)':<26}" + "".join(f"{retro_scores[p]:>13.4f}" for p in ORDER))
    print(f"{'post-specialisation':<26}" + "".join(f"{post_scores[p]:>13.4f}" for p in ORDER))
    print()
    print("change against stock BERT:")
    print(f"{'condition':<26}" + "".join(f"{p:>13}" for p in ORDER))
    print(f"{'retrofitting':<26}"
          + "".join(f"{retro_scores[p] - base[p]:>+13.4f}" for p in ORDER))
    print(f"{'post-specialisation':<26}"
          + "".join(f"{post_scores[p] - base[p]:>+13.4f}" for p in ORDER))
    print(f"{'RSR (for reference)':<26}"
          + "".join(f"{v:>13}" for v in ["+0.053", "-0.008", "+0.045", "+0.091"]))
    print()
    print("The row that matters is 'neither'. Retrofitting scores exactly +0.0000")
    print("there because those words have no edges. If post-specialisation is also")
    print("near zero, the learned map did not extrapolate and the RSR claim stands")
    print("as stated. If it is large, the honest claim narrows to a comparison")
    print("between two ways of reaching unsupervised words.")


if __name__ == "__main__":
    main()
