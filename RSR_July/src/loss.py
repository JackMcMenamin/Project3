"""
Loss Functions Module for Representational Similarity Regularisation (RSR)

This module implements the differentiable Soft-Spearman rank correlation loss.
Because standard rank operations are non-differentiable (they rely on sorting),
this module uses a temperature-scaled pairwise logistic sigmoid to approximate 
rank order, allowing us to backpropagate human similarity topologies directly 
into the language model.

Workflow Context:
- Consumed by: `train.py`. Called during the RSR forward pass every 4th step.
- Input: Takes 768-d vectors from `RobertaRSRModel.target_vectors()` (standardisation disabled by default).
- Output: Computes the loss `(1 - soft_spearman)` which is backpropagated to update the model.

Functions:
- soft_rank: Differentiable approximation of rank positions.
- soft_spearman: Differentiable approximation of Spearman correlation.
- rsr_loss_for_batch: Computes the full RSR loss given model embeddings and human scores.
"""

import torch
import torch.nn.functional as F

def soft_rank(x, tau=0.1):
    """
    Computes a differentiable soft rank of a 1D tensor using pairwise comparisons.
    
    Instead of hard-sorting, this function evaluates the logistic sigmoid of the 
    difference between all pairs of elements. If x_j > x_i, the sigmoid approaches 1. 
    Summing these pairwise indicator approximations yields a continuous, differentiable 
    rank for each element.
    
    Args:
        x (Tensor): 1D tensor of values to rank. Shape: (N,)
        tau (float): Temperature parameter controlling the steepness of the sigmoid.
                     Lower tau approaches a hard step function.
                     
    Returns:
        Tensor: The soft-rank values of the input tensor. Shape: (N,)
    """
    x_i = x.unsqueeze(1)
    x_j = x.unsqueeze(0)
    
    # Calculate pairwise differences
    diff = x_j - x_i
    
    # Scale tau dynamically based on the standard deviation of the input distribution.
    # This is critical because RoBERTa cosine similarities are highly compressed (small std)
    # compared to human scores, which causes a fixed tau to degenerate into a linear
    # transformation rather than a logistic rank approximation.
    # We MUST detach the standard deviation to prevent the model from maliciously manipulating
    # batch variance to minimize the loss, and we MUST add epsilon before sqrt to avoid NaN gradients.
    std_x = torch.sqrt(x.var(unbiased=False) + 1e-8).detach()
    scaled_tau = tau * torch.clamp(std_x, min=1e-6)
    
    # Logistic sigmoid approximation of the indicator function I(x_j > x_i)
    pairwise_cmp = torch.sigmoid(diff / scaled_tau)
    
    # Sum over j to get the rank (adding 1 to mimic 1-based indexing)
    rank = pairwise_cmp.sum(dim=1) + 1.0
    return rank

def soft_spearman(x, y, tau=0.1):
    """
    Computes a differentiable Spearman rank correlation coefficient.
    
    This function converts raw values into soft ranks, then calculates the standard 
    Pearson correlation coefficient between the two sets of ranks.
    
    Args:
        x (Tensor): First 1D tensor (e.g., predicted cosine similarities).
        y (Tensor): Second 1D tensor (e.g., human similarity scores).
        tau (float): Temperature for the soft ranking function.
        
    Returns:
        Tensor: A scalar tensor representing the correlation [-1.0, 1.0].
    """
    # Convert raw scores to soft ranks
    rx = soft_rank(x, tau)
    ry = soft_rank(y, tau)
    
    # Calculate means of the ranks
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    
    # Center the ranks around 0
    rx_centered = rx - rx_mean
    ry_centered = ry - ry_mean
    
    # Calculate covariance and variances
    cov = (rx_centered * ry_centered).sum()
    var_rx = (rx_centered ** 2).sum()
    var_ry = (ry_centered ** 2).sum()
    
    # Compute Pearson correlation (covariance / product of standard deviations)
    corr = cov / torch.sqrt(var_rx * var_ry + 1e-8)
    return corr

def rsr_loss_for_batch(vectors, human_scores_matrix, valid_pairs_mask, tau=0.1):
    """
    Computes the Representational Similarity Regularisation (RSR) loss for a batch.
    
    Given N target vectors, this function computes all N(N-1)/2 unique 
    pairwise cosine similarities. It extracts the valid pairs utilizing highly-optimized 
    upper-triangular PyTorch boolean masking. It then correlates these predicted similarities 
    against the known human judgements using the soft Spearman function.
    
    Args:
        vectors (Tensor): The semantic embeddings. Shape: (N, embed_dim).
        human_scores_matrix (Tensor): Matrix of human scores. Shape: (N, N).
        valid_pairs_mask (Tensor): Boolean mask indicating which pairs have a 
                                   known human similarity score. Shape: (N, N).
        tau (float): Temperature parameter for soft ranking.
        
    Returns:
        Tensor: A scalar loss value computed as (1.0 - soft_spearman_correlation).
    """
    N = vectors.size(0)
    
    # L2 normalize vectors to compute cosine similarity via dot product
    normed_vectors = F.normalize(vectors, p=2, dim=1)
    
    # Compute the full NxN cosine similarity matrix
    sim_matrix = torch.matmul(normed_vectors, normed_vectors.t())
    
    # --- Vectorized PyTorch Approach ---
    upper_tri_mask = torch.triu(valid_pairs_mask, diagonal=1)
    pred_sims = sim_matrix[upper_tri_mask]
    target_sims = human_scores_matrix[upper_tri_mask]
    
    # If there are fewer than 2 valid pairs, correlation cannot be computed
    if len(pred_sims) < 2:
        return torch.tensor(0.0, requires_grad=True, device=vectors.device)
        
    # Compute soft Spearman correlation and invert it to create a loss function
    rho_soft = soft_spearman(pred_sims, target_sims, tau)
    loss = 1.0 - rho_soft
    
    return loss
