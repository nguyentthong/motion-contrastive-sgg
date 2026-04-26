"""Optimal Transport for Mask Tube Relation Quantification.

Implements the Sinkhorn-based optimal transport distance between two mask tube
representations, as described in Section "Optimal Transport for Mask Tube Relation
Quantification" of the paper.

Key equations:
    d_OT = min_{T in Pi(a,b)} sum_k sum_l T_{k,l} * c(h_{i,k}, h_{j,l})    (Eq. 7)
    sim(h^a, h^p) = alpha - d_OT                                              (Eq. 9)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class OTConfig:
    """Configuration for optimal transport computation."""
    alpha: float = 10.0          # Margin for converting distance to similarity (Eq. 9)
    n_iter: int = 1000           # Max Sinkhorn iterations (Algorithm 1)
    tau: float = 0.05            # Entropic regularization temperature
    stop_threshold: float = 1e-6 # Early stopping threshold for Sinkhorn


def cosine_distance_matrix(h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine distance cost matrix.

    c(h_{i,k}, h_{j,l}) = 1 - (h_{i,k} . h_{j,l}) / (||h_{i,k}|| * ||h_{j,l}||)

    Args:
        h_i: Tensor of shape [T_i, D] — support of distribution mu.
        h_j: Tensor of shape [T_j, D] — support of distribution nu.

    Returns:
        Cost matrix C of shape [T_i, T_j].
    """
    h_i_norm = F.normalize(h_i, p=2, dim=-1)
    h_j_norm = F.normalize(h_j, p=2, dim=-1)
    cosine_sim = torch.mm(h_i_norm, h_j_norm.t())  # [T_i, T_j]
    return 1.0 - cosine_sim


def sinkhorn_ot_distance(
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    config: OTConfig | None = None,
) -> torch.Tensor:
    """Compute the optimal transport distance between two mask tube representations.

    Implements Algorithm 1 from the paper. Treats H_i and H_j as discrete distributions
    mu = sum_k a_k * delta_{h_{i,k}} and nu = sum_l b_l * delta_{h_{j,l}}
    with uniform weights a = 1/T_i and b = 1/T_j.

    Args:
        h_i: Tensor of shape [T_i, D] — tube representation of entity i.
        h_j: Tensor of shape [T_j, D] — tube representation of entity j.
        config: OT configuration. Uses defaults if None.

    Returns:
        Scalar tensor — the optimal transport distance d_OT.
    """
    if config is None:
        config = OTConfig()

    T_i, D = h_i.shape
    T_j, _ = h_j.shape
    device = h_i.device

    # Cost matrix C_{k,l} = c(h_{i,k}, h_{j,l})
    C = cosine_distance_matrix(h_i, h_j)  # [T_i, T_j]

    # Uniform marginals
    a = torch.ones(T_i, device=device) / T_i
    b = torch.ones(T_j, device=device) / T_j

    best_d_ot = torch.tensor(float("inf"), device=device)

    s_max = min(T_i, T_j)

    # Algorithm 1: iterate over partial transport masses s = 1..min(T_i, T_j)
    # For efficiency in practice, the full transport (s_max) is the main case.
    # We implement the loop but typically s_max is the dominant solution.
    for s in range(1, s_max + 1):
        # Initialize transport plan via Gibbs kernel
        T_plan = torch.exp(-C / config.tau)  # [T_i, T_j]

        # Scale to have total mass s
        total_mass = T_plan.sum()
        if total_mass > 0:
            T_plan = (s / total_mass) * T_plan

        # Sinkhorn iterations
        for _ in range(config.n_iter):
            # Row scaling: pa = min(a / (T @ 1_{T_j}), 1_{T_i})
            row_sum = T_plan.sum(dim=1)  # [T_i]
            pa = torch.min(a / (row_sum + 1e-10), torch.ones_like(a))
            T_a = pa.unsqueeze(1) * T_plan  # diag(pa) @ T

            # Column scaling: pb = min(b / (T_a^T @ 1_{T_i}), 1_{T_j})
            col_sum = T_a.sum(dim=0)  # [T_j]
            pb = torch.min(b / (col_sum + 1e-10), torch.ones_like(b))
            T_b = T_a * pb.unsqueeze(0)  # T_a @ diag(pb)

            # Re-scale to mass s
            total = T_b.sum()
            if total > 0:
                T_plan = (s / total) * T_b
            else:
                break

        # Compute transport cost
        d_ot_s = (T_plan * C).sum()
        best_d_ot = torch.min(best_d_ot, d_ot_s)

    return best_d_ot


def ot_similarity(
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    config: OTConfig | None = None,
) -> torch.Tensor:
    """Convert OT distance to similarity: sim = alpha - d_OT (Eq. 9).

    Args:
        h_i: Tensor [T_i, D].
        h_j: Tensor [T_j, D].
        config: OT configuration.

    Returns:
        Scalar similarity value.
    """
    if config is None:
        config = OTConfig()
    d_ot = sinkhorn_ot_distance(h_i, h_j, config)
    return config.alpha - d_ot


def batch_ot_similarity(
    anchors: list[torch.Tensor],
    others: list[torch.Tensor],
    config: OTConfig | None = None,
) -> torch.Tensor:
    """Compute OT similarity for a batch of tube pairs.

    Args:
        anchors: List of tensors, each [T_a, D].
        others: List of tensors, each [T_o, D].
        config: OT configuration.

    Returns:
        Tensor of shape [batch_size] with similarity values.
    """
    sims = []
    for a, o in zip(anchors, others):
        sims.append(ot_similarity(a, o, config))
    return torch.stack(sims)


# ---- Ablation alternatives (Section: Ablation Study) ----

def pooling_cosine_similarity(h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
    """Ablation: temporal mean pooling + cosine similarity."""
    pooled_i = h_i.mean(dim=0)  # [D]
    pooled_j = h_j.mean(dim=0)  # [D]
    return F.cosine_similarity(pooled_i.unsqueeze(0), pooled_j.unsqueeze(0)).squeeze()


def pooling_l2_similarity(
    h_i: torch.Tensor, h_j: torch.Tensor, alpha: float = 10.0
) -> torch.Tensor:
    """Ablation: temporal mean pooling + negative L2 distance."""
    pooled_i = h_i.mean(dim=0)
    pooled_j = h_j.mean(dim=0)
    return alpha - torch.norm(pooled_i - pooled_j, p=2)


def get_similarity_fn(method: str, config: OTConfig | None = None):
    """Factory for similarity functions.

    Args:
        method: One of 'optimal_transport', 'pooling_cosine', 'pooling_l2'.
        config: OT config (used only for optimal_transport).

    Returns:
        A callable (h_i, h_j) -> scalar similarity.
    """
    if method == "optimal_transport":
        cfg = config or OTConfig()
        return lambda h_i, h_j: ot_similarity(h_i, h_j, cfg)
    elif method == "pooling_cosine":
        return pooling_cosine_similarity
    elif method == "pooling_l2":
        alpha = config.alpha if config else 10.0
        return lambda h_i, h_j: pooling_l2_similarity(h_i, h_j, alpha)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
