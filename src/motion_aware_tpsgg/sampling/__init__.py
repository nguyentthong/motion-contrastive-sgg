"""Positive and negative sampling strategies for contrastive learning.

Implements:
    - Positive sampling: same subject-relation-object category from different video
    - Shuffle-based negative: temporal permutations of anchor tube (Eq. 6)
    - Triplet-based negative: hard negatives from same video with multinomial sampling
    - Motion strength estimation via optical flow edges
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Triplet:
    """A subject-relation-object triplet with its tube representations."""
    video_id: str
    subject_cat: int
    relation_cat: int
    object_cat: int
    subject_tube: torch.Tensor  # [T, D]
    object_tube: torch.Tensor   # [T, D]
    anchor_repr: torch.Tensor | None = None  # [T, 2D] = [H_sub, H_obj] (Eq. 4)
    motion_strength: float = 0.0

    def build_anchor(self) -> torch.Tensor:
        """H^a_{i,j} = [H^{sub}_i, H^{obj}_j] (Eq. 4)."""
        self.anchor_repr = torch.cat([self.subject_tube, self.object_tube], dim=-1)
        return self.anchor_repr

    @property
    def category_key(self) -> tuple[int, int, int]:
        return (self.subject_cat, self.relation_cat, self.object_cat)


# ==================== Positive Sampling ====================

class PositiveSampler:
    """Sample positive triplets: same (subject, relation, object) category from different video.

    From the paper: "we extract mask tube representations from the entities of the same
    subject and object category that exhibit a similar groundtruth relation from another video."
    """

    def __init__(self, triplets: list[Triplet]):
        """Build an index of triplets grouped by category key and video."""
        self.category_index: dict[tuple[int, int, int], dict[str, list[Triplet]]] = {}
        for t in triplets:
            key = t.category_key
            if key not in self.category_index:
                self.category_index[key] = {}
            vid = t.video_id
            if vid not in self.category_index[key]:
                self.category_index[key][vid] = []
            self.category_index[key][vid].append(t)

    def sample(self, anchor: Triplet) -> Triplet | None:
        """Sample a positive triplet from a different video with same category.

        Returns None if no valid positive exists.
        """
        key = anchor.category_key
        if key not in self.category_index:
            return None

        candidates = []
        for vid, triplets in self.category_index[key].items():
            if vid != anchor.video_id:
                candidates.extend(triplets)

        if not candidates:
            return None
        return random.choice(candidates)


# ==================== Shuffle-based Negative Sampling ====================

def shuffle_tube(tube: torch.Tensor) -> torch.Tensor:
    """Create a negative by temporally shuffling the tube (Eq. 6): H^n = pi(H^a).

    Args:
        tube: [T, D] tube representation.

    Returns:
        Shuffled tube of same shape.
    """
    T = tube.shape[0]
    perm = torch.randperm(T)
    return tube[perm]


def estimate_motion_strength(
    flow_magnitude: np.ndarray | None = None,
    entity_masks: np.ndarray | None = None,
) -> float:
    """Estimate motion strength of a mask tube using optical flow edges.

    From paper: "We estimate flow edges via employing a Sobel filter onto the flow
    magnitude map and take the median over the flow edge pixels of the entity masks.
    Then, we select mask tubes whose the maximum value across the optical flow surpasses
    a threshold gamma."

    Args:
        flow_magnitude: [T-1, H, W] optical flow magnitude between consecutive frames.
            If None, returns 0.0 (for synthetic/testing).
        entity_masks: [T, H, W] binary masks for the entity.

    Returns:
        Maximum motion strength value across frames.
    """
    if flow_magnitude is None or entity_masks is None:
        return 0.0

    try:
        import cv2
    except ImportError:
        # Fallback: rough estimate without Sobel
        max_motion = 0.0
        for t in range(flow_magnitude.shape[0]):
            mask = entity_masks[t + 1] if (t + 1) < entity_masks.shape[0] else entity_masks[t]
            masked_flow = flow_magnitude[t] * mask
            if mask.sum() > 0:
                median_val = float(np.median(masked_flow[mask > 0]))
                max_motion = max(max_motion, median_val)
        return max_motion

    max_motion = 0.0
    for t in range(flow_magnitude.shape[0]):
        # Apply Sobel filter to flow magnitude
        flow_t = flow_magnitude[t].astype(np.float32)
        sobel_x = cv2.Sobel(flow_t, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(flow_t, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Take median over entity mask pixels
        mask = entity_masks[min(t + 1, entity_masks.shape[0] - 1)]
        if mask.sum() > 0:
            median_edge = float(np.median(edge_magnitude[mask > 0]))
            max_motion = max(max_motion, median_edge)

    return max_motion


class ShuffleNegativeSampler:
    """Generate shuffle-based negatives for strong-motion tubes.

    Only applies to tubes whose motion strength exceeds gamma (default 9.0).
    """

    def __init__(self, gamma: float = 9.0, num_negatives: int = 3):
        self.gamma = gamma
        self.num_negatives = num_negatives

    def is_strong_motion(self, triplet: Triplet) -> bool:
        return triplet.motion_strength > self.gamma

    def sample(self, anchor: Triplet) -> list[torch.Tensor]:
        """Generate shuffle-based negatives for anchor if it has strong motion.

        Returns empty list if motion is below threshold.
        """
        if not self.is_strong_motion(anchor):
            return []

        anchor_repr = anchor.anchor_repr
        if anchor_repr is None:
            anchor_repr = anchor.build_anchor()

        negatives = []
        for _ in range(self.num_negatives):
            shuffled = shuffle_tube(anchor_repr)
            negatives.append(shuffled)
        return negatives


# ==================== Triplet-based Negative Sampling ====================

class TripletNegativeSampler:
    """Sample negative triplets from the same video, preferring hard negatives.

    From paper: "We create a multi-nomial distribution, where triplets that share more
    subject, relation, or object categories with the anchor will be more likely to be drawn."
    """

    def __init__(
        self,
        num_negatives: int = 3,
        shared_category_weight: float = 2.0,
    ):
        self.num_negatives = num_negatives
        self.shared_category_weight = shared_category_weight

    def _compute_sharing_score(self, anchor: Triplet, candidate: Triplet) -> float:
        """Count how many components (subject, relation, object) are shared."""
        score = 0.0
        if anchor.subject_cat == candidate.subject_cat:
            score += 1.0
        if anchor.relation_cat == candidate.relation_cat:
            score += 1.0
        if anchor.object_cat == candidate.object_cat:
            score += 1.0
        return score

    def sample(
        self, anchor: Triplet, same_video_triplets: list[Triplet]
    ) -> list[torch.Tensor]:
        """Sample hard negative triplets from same video with multinomial distribution.

        Args:
            anchor: The anchor triplet.
            same_video_triplets: All triplets from the same video.

        Returns:
            List of negative tube representations [T, 2D].
        """
        # Filter out the anchor itself
        candidates = [
            t for t in same_video_triplets
            if t.category_key != anchor.category_key
        ]
        if not candidates:
            return []

        # Build multinomial distribution based on category sharing
        weights = []
        for c in candidates:
            score = self._compute_sharing_score(anchor, c)
            # Higher score = more shared categories = harder negative = higher weight
            w = self.shared_category_weight ** score
            weights.append(w)

        weights_t = torch.tensor(weights, dtype=torch.float32)
        weights_t = weights_t / weights_t.sum()

        num_to_sample = min(self.num_negatives, len(candidates))
        indices = torch.multinomial(weights_t, num_to_sample, replacement=len(candidates) < num_to_sample)

        negatives = []
        for idx in indices:
            c = candidates[idx.item()]
            neg_repr = c.anchor_repr if c.anchor_repr is not None else c.build_anchor()
            negatives.append(neg_repr)

        return negatives
