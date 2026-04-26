"""Contrastive and classification losses for motion-aware TPSGG.

Implements:
    - InfoNCE contrastive loss (Eq. 5)
    - Relation cross-entropy loss (Eq. 3)
    - Combined training loss

Key equation (Eq. 5):
    L_cont = -log( exp(sim(H^a, H^p)) / (exp(sim(H^a, H^p)) + sum_z exp(sim(H^a, H^n_z))) )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from motion_aware_tpsgg.transport import OTConfig, get_similarity_fn


@dataclass
class LossConfig:
    """Configuration for all losses."""
    # Contrastive loss weights
    weight_relation_ce: float = 1.0
    weight_shuffle_contrastive: float = 1.0
    weight_triplet_contrastive: float = 1.0

    # Similarity method: 'optimal_transport', 'pooling_cosine', 'pooling_l2'
    similarity_method: str = "optimal_transport"

    # OT config
    ot_alpha: float = 10.0
    ot_n_iter: int = 1000
    ot_tau: float = 0.05

    # Motion threshold for shuffle-based contrastive
    gamma: float = 9.0

    # Temperature for InfoNCE (if needed beyond OT similarity)
    temperature: float = 1.0


def info_nce_loss(
    sim_positive: torch.Tensor,
    sim_negatives: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """InfoNCE contrastive loss (Eq. 5).

    L_cont = -log( exp(sim(H^a, H^p)) / (exp(sim(H^a, H^p)) + sum_z exp(sim(H^a, H^n_z))) )

    Args:
        sim_positive: Scalar tensor — similarity between anchor and positive.
        sim_negatives: Tensor [N_n] — similarities between anchor and each negative.
        temperature: Scaling temperature.

    Returns:
        Scalar loss tensor.
    """
    sim_pos_scaled = sim_positive / temperature
    sim_neg_scaled = sim_negatives / temperature

    # Numerator: exp(sim(H^a, H^p))
    # Denominator: exp(sim(H^a, H^p)) + sum_z exp(sim(H^a, H^n_z))
    logits = torch.cat([sim_pos_scaled.unsqueeze(0), sim_neg_scaled], dim=0)
    # Target: index 0 is the positive
    log_prob = F.log_softmax(logits, dim=0)
    loss = -log_prob[0]
    return loss


def batch_info_nce_loss(
    sim_positives: torch.Tensor,
    sim_negatives_list: list[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Batch InfoNCE loss over multiple anchors.

    Args:
        sim_positives: [B] similarities for each anchor-positive pair.
        sim_negatives_list: List of B tensors, each [N_n_i] negatives per anchor.
        temperature: Scaling temperature.

    Returns:
        Mean loss over the batch.
    """
    losses = []
    for i in range(sim_positives.shape[0]):
        loss_i = info_nce_loss(sim_positives[i], sim_negatives_list[i], temperature)
        losses.append(loss_i)
    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


class RelationCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for relation classification (after Eq. 3)."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_classes = num_classes

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] relation logits.
            targets: [B] integer class labels.
        """
        return self.ce(logits, targets)


class MotionAwareContrastiveLoss(nn.Module):
    """Combined loss: relation CE + shuffle-based contrastive + triplet-based contrastive.

    Total loss = w_ce * L_ce + w_shuffle * L_shuffle + w_triplet * L_triplet
    """

    def __init__(self, config: LossConfig, num_relation_classes: int):
        super().__init__()
        self.config = config
        self.relation_ce = RelationCrossEntropyLoss(num_relation_classes)

        ot_config = OTConfig(
            alpha=config.ot_alpha,
            n_iter=config.ot_n_iter,
            tau=config.ot_tau,
        )
        self.sim_fn = get_similarity_fn(config.similarity_method, ot_config)

    def compute_shuffle_contrastive(
        self,
        anchor_tubes: list[torch.Tensor],
        positive_tubes: list[torch.Tensor],
        negative_tubes_list: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Shuffle-based contrastive loss.

        Args:
            anchor_tubes: List of B anchor representations, each [T, 2D].
            positive_tubes: List of B positive representations.
            negative_tubes_list: List of B lists, each containing N_n negatives.
        """
        if not anchor_tubes:
            return torch.tensor(0.0, requires_grad=True)

        sim_pos = []
        sim_neg_list = []
        for a, p, negs in zip(anchor_tubes, positive_tubes, negative_tubes_list):
            sim_pos.append(self.sim_fn(a, p))
            sim_neg = torch.stack([self.sim_fn(a, n) for n in negs]) if negs else torch.tensor([], device=a.device)
            sim_neg_list.append(sim_neg)

        sim_pos_t = torch.stack(sim_pos)
        return batch_info_nce_loss(sim_pos_t, sim_neg_list, self.config.temperature)

    def compute_triplet_contrastive(
        self,
        anchor_tubes: list[torch.Tensor],
        positive_tubes: list[torch.Tensor],
        negative_tubes_list: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Triplet-based contrastive loss. Same structure as shuffle but different negatives."""
        return self.compute_shuffle_contrastive(anchor_tubes, positive_tubes, negative_tubes_list)

    def forward(
        self,
        relation_logits: torch.Tensor,
        relation_targets: torch.Tensor,
        shuffle_data: dict | None = None,
        triplet_data: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            relation_logits: [B, C] logits from relation classifier.
            relation_targets: [B] ground-truth relation classes.
            shuffle_data: Dict with 'anchors', 'positives', 'negatives' lists of tensors.
            triplet_data: Dict with 'anchors', 'positives', 'negatives' lists of tensors.

        Returns:
            Dict with 'total', 'relation_ce', 'shuffle_contrastive', 'triplet_contrastive'.
        """
        loss_ce = self.relation_ce(relation_logits, relation_targets)
        total = self.config.weight_relation_ce * loss_ce

        loss_shuffle = torch.tensor(0.0, device=relation_logits.device)
        if shuffle_data is not None and shuffle_data.get("anchors"):
            loss_shuffle = self.compute_shuffle_contrastive(
                shuffle_data["anchors"],
                shuffle_data["positives"],
                shuffle_data["negatives"],
            )
            total = total + self.config.weight_shuffle_contrastive * loss_shuffle

        loss_triplet = torch.tensor(0.0, device=relation_logits.device)
        if triplet_data is not None and triplet_data.get("anchors"):
            loss_triplet = self.compute_triplet_contrastive(
                triplet_data["anchors"],
                triplet_data["positives"],
                triplet_data["negatives"],
            )
            total = total + self.config.weight_triplet_contrastive * loss_triplet

        return {
            "total": total,
            "relation_ce": loss_ce,
            "shuffle_contrastive": loss_shuffle,
            "triplet_contrastive": loss_triplet,
        }
