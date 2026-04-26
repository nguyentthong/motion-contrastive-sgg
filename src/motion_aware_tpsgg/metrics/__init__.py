"""Evaluation metrics for temporal panoptic scene graph generation.

Implements:
    - Volume Intersection over Union (vIoU) for mask tubes
    - Recall@K (R@K) at K=20, 50, 100
    - Mean Recall@K (mR@K) at K=20, 50, 100
    - Support for vIoU thresholds 0.5 and 0.1

A predicted triplet is correct if:
    1) Subject category is correct
    2) Object category is correct
    3) Predicate/relation category is correct
    4) Both predicted subject and object mask tubes meet the vIoU threshold with GT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch


def compute_viou(
    pred_tube: torch.Tensor | np.ndarray,
    gt_tube: torch.Tensor | np.ndarray,
) -> float:
    """Compute Volume Intersection over Union between two mask tubes.

    Args:
        pred_tube: [T, H, W] binary predicted mask tube.
        gt_tube: [T, H, W] binary ground-truth mask tube.

    Returns:
        vIoU value in [0, 1].
    """
    if isinstance(pred_tube, torch.Tensor):
        pred_tube = pred_tube.cpu().numpy()
    if isinstance(gt_tube, torch.Tensor):
        gt_tube = gt_tube.cpu().numpy()

    pred_bool = pred_tube.astype(bool)
    gt_bool = gt_tube.astype(bool)

    # Handle shape mismatch by taking the minimum temporal length
    T = min(pred_bool.shape[0], gt_bool.shape[0])
    pred_bool = pred_bool[:T]
    gt_bool = gt_bool[:T]

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


@dataclass
class PredictedTriplet:
    """A predicted scene graph triplet."""
    subject_cat: int
    object_cat: int
    relation_cat: int
    score: float  # Confidence score for ranking
    subject_tube: np.ndarray | torch.Tensor | None = None  # [T, H, W]
    object_tube: np.ndarray | torch.Tensor | None = None   # [T, H, W]


@dataclass
class GroundTruthTriplet:
    """A ground-truth scene graph triplet."""
    subject_cat: int
    object_cat: int
    relation_cat: int
    subject_tube: np.ndarray | torch.Tensor | None = None  # [T, H, W]
    object_tube: np.ndarray | torch.Tensor | None = None   # [T, H, W]


def _match_triplet(
    pred: PredictedTriplet,
    gt: GroundTruthTriplet,
    viou_threshold: float,
) -> bool:
    """Check if a predicted triplet matches a ground-truth triplet."""
    # Check category match
    if pred.subject_cat != gt.subject_cat:
        return False
    if pred.object_cat != gt.object_cat:
        return False
    if pred.relation_cat != gt.relation_cat:
        return False

    # Check vIoU if tubes available
    if pred.subject_tube is not None and gt.subject_tube is not None:
        sub_viou = compute_viou(pred.subject_tube, gt.subject_tube)
        if sub_viou < viou_threshold:
            return False

    if pred.object_tube is not None and gt.object_tube is not None:
        obj_viou = compute_viou(pred.object_tube, gt.object_tube)
        if obj_viou < viou_threshold:
            return False

    return True


def recall_at_k(
    predictions: list[PredictedTriplet],
    ground_truths: list[GroundTruthTriplet],
    k: int,
    viou_threshold: float = 0.5,
) -> float:
    """Compute Recall@K for a single video/sample.

    Takes top-K predictions by score, counts how many GT triplets are recalled.

    Args:
        predictions: List of predicted triplets.
        ground_truths: List of ground-truth triplets.
        k: Number of top predictions to consider.
        viou_threshold: Minimum vIoU for tube matching.

    Returns:
        Recall value in [0, 1].
    """
    if not ground_truths:
        return 0.0

    # Sort predictions by score descending, take top K
    sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)[:k]

    gt_matched = [False] * len(ground_truths)

    for pred in sorted_preds:
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            if _match_triplet(pred, gt, viou_threshold):
                gt_matched[gt_idx] = True
                break

    num_recalled = sum(gt_matched)
    return num_recalled / len(ground_truths)


def mean_recall_at_k(
    all_predictions: list[list[PredictedTriplet]],
    all_ground_truths: list[list[GroundTruthTriplet]],
    k: int,
    viou_threshold: float = 0.5,
    num_relation_classes: int | None = None,
) -> float:
    """Compute Mean Recall@K: average R@K per relation class, then average across classes.

    Args:
        all_predictions: List of predictions per sample.
        all_ground_truths: List of ground-truths per sample.
        k: K value.
        viou_threshold: vIoU threshold.
        num_relation_classes: If provided, average over this many classes.

    Returns:
        mR@K value.
    """
    # Group ground-truths and predictions by relation class
    per_class_preds: dict[int, list[PredictedTriplet]] = defaultdict(list)
    per_class_gts: dict[int, list[GroundTruthTriplet]] = defaultdict(list)

    for preds, gts in zip(all_predictions, all_ground_truths):
        for gt in gts:
            per_class_gts[gt.relation_cat].append(gt)
        for pred in preds:
            per_class_preds[pred.relation_cat].append(pred)

    # Compute per-class recall
    all_classes = set(per_class_gts.keys())
    if not all_classes:
        return 0.0

    class_recalls = []
    for cls in all_classes:
        cls_gts = per_class_gts[cls]
        cls_preds = per_class_preds.get(cls, [])

        # Sort by score, take top-K
        sorted_preds = sorted(cls_preds, key=lambda p: p.score, reverse=True)[:k]

        # Count matched GTs
        gt_matched = [False] * len(cls_gts)
        for pred in sorted_preds:
            for gt_idx, gt in enumerate(cls_gts):
                if gt_matched[gt_idx]:
                    continue
                if _match_triplet(pred, gt, viou_threshold):
                    gt_matched[gt_idx] = True
                    break

        if cls_gts:
            class_recalls.append(sum(gt_matched) / len(cls_gts))

    if not class_recalls:
        return 0.0
    return float(np.mean(class_recalls))


@dataclass
class EvaluationResult:
    """Full evaluation results."""
    viou_threshold: float
    r_at_20: float = 0.0
    r_at_50: float = 0.0
    r_at_100: float = 0.0
    mr_at_20: float = 0.0
    mr_at_50: float = 0.0
    mr_at_100: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            f"R@20 (vIoU={self.viou_threshold})": self.r_at_20,
            f"R@50 (vIoU={self.viou_threshold})": self.r_at_50,
            f"R@100 (vIoU={self.viou_threshold})": self.r_at_100,
            f"mR@20 (vIoU={self.viou_threshold})": self.mr_at_20,
            f"mR@50 (vIoU={self.viou_threshold})": self.mr_at_50,
            f"mR@100 (vIoU={self.viou_threshold})": self.mr_at_100,
        }

    def __str__(self) -> str:
        lines = [f"vIoU threshold = {self.viou_threshold}"]
        for key, val in self.to_dict().items():
            lines.append(f"  {key}: {val:.4f}")
        return "\n".join(lines)


def evaluate_full(
    all_predictions: list[list[PredictedTriplet]],
    all_ground_truths: list[list[GroundTruthTriplet]],
    viou_threshold: float = 0.5,
    num_relation_classes: int | None = None,
) -> EvaluationResult:
    """Run full evaluation at a given vIoU threshold.

    Args:
        all_predictions: Per-sample predictions.
        all_ground_truths: Per-sample ground-truths.
        viou_threshold: vIoU threshold.
        num_relation_classes: Optional, for mR@K.

    Returns:
        EvaluationResult with all metrics.
    """
    result = EvaluationResult(viou_threshold=viou_threshold)

    # R@K: average over all samples
    r_at = {}
    for k_val in [20, 50, 100]:
        sample_recalls = []
        for preds, gts in zip(all_predictions, all_ground_truths):
            r = recall_at_k(preds, gts, k_val, viou_threshold)
            sample_recalls.append(r)
        r_at[k_val] = float(np.mean(sample_recalls)) if sample_recalls else 0.0

    result.r_at_20 = r_at[20]
    result.r_at_50 = r_at[50]
    result.r_at_100 = r_at[100]

    # mR@K
    for k_val in [20, 50, 100]:
        mr = mean_recall_at_k(
            all_predictions, all_ground_truths, k_val, viou_threshold, num_relation_classes
        )
        if k_val == 20:
            result.mr_at_20 = mr
        elif k_val == 50:
            result.mr_at_50 = mr
        else:
            result.mr_at_100 = mr

    return result
