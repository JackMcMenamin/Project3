"""The RSR term: 1 minus a differentiable Spearman correlation between the
model's pairwise cosines and the human ratings."""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F

try:
    import torchsort
    HAS_TORCHSORT = True
except ImportError:
    HAS_TORCHSORT = False


def soft_rank_sigmoid(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Fallback soft rank for when torchsort is missing. The rank of x[i] is
    one plus the number of values below it; replacing that count with a sum of
    sigmoids makes it differentiable. Higher strength is closer to hard ranks.

    At strength 1.0 on cosines and [0, 1] ratings the differences are small, the
    sigmoid is nearly linear, and the result is close to Pearson correlation.
    All runs so far used this path at strength 1.0."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    ranks = torch.sigmoid(diff * strength).sum(dim=2) + 0.5
    return ranks.squeeze(0)


def soft_spearman(pred: torch.Tensor, target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Pearson correlation of soft ranks."""
    if HAS_TORCHSORT:
        pred_r = torchsort.soft_rank(pred.unsqueeze(0), regularization_strength=strength).squeeze(0)
        tgt_r = torchsort.soft_rank(target.unsqueeze(0), regularization_strength=strength).squeeze(0)
    else:
        pred_r = soft_rank_sigmoid(pred, strength)
        tgt_r = soft_rank_sigmoid(target, strength)
    pred_c = pred_r - pred_r.mean()
    tgt_c = tgt_r - tgt_r.mean()
    return (pred_c * tgt_c).mean() / (pred_c.std() * tgt_c.std() + 1e-8)


def pair_loss(vecs: torch.Tensor, words: list[str], sim_lookup: dict,
              min_pairs: int, strength: float) -> torch.Tensor | None:
    """RSR loss for one batch of word vectors.

    Every pair of words in the batch that has a human score contributes;
    unscored pairs are skipped. Returns None when fewer than `min_pairs` are
    scored, since the rank correlation is then too noisy to train on.
    """
    model_sims, human_sims = [], []
    for i, j in combinations(range(len(words)), 2):
        score = sim_lookup.get((words[i], words[j]))
        if score is None:
            continue
        model_sims.append(F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)))
        human_sims.append(score)
    if len(model_sims) < min_pairs:
        return None
    rho = soft_spearman(torch.cat(model_sims),
                        torch.tensor(human_sims, device=vecs.device, dtype=torch.float32),
                        strength=strength)
    return 1.0 - rho
