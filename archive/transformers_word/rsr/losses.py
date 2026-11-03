"""
Differentiable Spearman loss used by RSR.

L_rsr = 1 - rho_soft(model_cosines, human_scores)

Uses `torchsort.soft_rank` when available (faster, lower memory); otherwise
falls back to a sigmoid-pairwise soft rank.
"""
from __future__ import annotations

import torch

try:
    import torchsort

    HAS_TORCHSORT = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCHSORT = False

DEFAULT_RANK_STRENGTH = 1.0


def soft_rank_custom(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """Sigmoid-pairwise soft rank — fallback when torchsort is not installed."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    diff = x.unsqueeze(2) - x.unsqueeze(1)
    soft_comparisons = torch.sigmoid(diff * regularization_strength)
    ranks = soft_comparisons.sum(dim=2) + 0.5
    return ranks.squeeze(0)


def soft_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    regularization_strength: float = DEFAULT_RANK_STRENGTH,
) -> torch.Tensor:
    """Differentiable Spearman correlation between two 1-D score vectors."""
    if HAS_TORCHSORT:
        pred_rank = torchsort.soft_rank(
            pred.unsqueeze(0), regularization_strength=regularization_strength
        ).squeeze(0)
        target_rank = torchsort.soft_rank(
            target.unsqueeze(0), regularization_strength=regularization_strength
        ).squeeze(0)
    else:
        pred_rank = soft_rank_custom(pred, regularization_strength)
        target_rank = soft_rank_custom(target, regularization_strength)

    pred_centered = pred_rank - pred_rank.mean()
    target_centered = target_rank - target_rank.mean()

    cov = (pred_centered * target_centered).mean()
    correlation = cov / (pred_centered.std() * target_centered.std() + 1e-8)
    return correlation
