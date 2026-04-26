"""Evaluation pipeline for TPSGG models.

Runs inference on a dataset and computes R@K, mR@K at various vIoU thresholds.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from motion_aware_tpsgg.datasets.base import (
    TPSGGDataset,
    TPSGGSample,
    collate_relation_pairs,
)
from motion_aware_tpsgg.models import TPSGGModel
from motion_aware_tpsgg.metrics import (
    PredictedTriplet,
    GroundTruthTriplet,
    evaluate_full,
    EvaluationResult,
)


def run_evaluation(
    model: TPSGGModel,
    dataset: TPSGGDataset,
    device: str = "cpu",
    pad_length: int = 16,
    viou_thresholds: list[float] | None = None,
    top_k_values: list[int] | None = None,
) -> list[EvaluationResult]:
    """Run full evaluation on a dataset.

    Args:
        model: Trained TPSGG model.
        dataset: Evaluation dataset.
        device: Device string.
        pad_length: Pad tube temporal dimension.
        viou_thresholds: List of vIoU thresholds (default [0.5, 0.1]).
        top_k_values: Unused here but reserved for future.

    Returns:
        List of EvaluationResult, one per vIoU threshold.
    """
    if viou_thresholds is None:
        viou_thresholds = [0.5, 0.1]

    model = model.to(device)
    model.eval()

    all_predictions: list[list[PredictedTriplet]] = []
    all_ground_truths: list[list[GroundTruthTriplet]] = []

    with torch.no_grad():
        for sample in dataset.samples:
            if sample.num_entities < 2:
                all_predictions.append([])
                all_ground_truths.append([])
                continue

            # Build ground-truth triplets
            gts = []
            for rel in sample.relations:
                if rel.subject_idx >= sample.num_entities or rel.object_idx >= sample.num_entities:
                    continue
                gt = GroundTruthTriplet(
                    subject_cat=sample.entity_labels[rel.subject_idx].item(),
                    object_cat=sample.entity_labels[rel.object_idx].item(),
                    relation_cat=rel.relation_cat,
                    subject_tube=sample.mask_tubes[rel.subject_idx] if sample.mask_tubes is not None else None,
                    object_tube=sample.mask_tubes[rel.object_idx] if sample.mask_tubes is not None else None,
                )
                gts.append(gt)

            # Generate predictions for all entity pairs
            preds = []
            tube_feats = sample.tube_features.to(device)  # [N, T, D]

            # Pad if needed
            T = tube_feats.shape[1]
            if T < pad_length:
                pad = torch.zeros(
                    tube_feats.shape[0], pad_length - T, tube_feats.shape[2],
                    device=tube_feats.device,
                )
                tube_feats = torch.cat([tube_feats, pad], dim=1)
            elif T > pad_length:
                tube_feats = tube_feats[:, :pad_length]

            N = sample.num_entities
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    sub_tube = tube_feats[i].unsqueeze(0)  # [1, T, D]
                    obj_tube = tube_feats[j].unsqueeze(0)  # [1, T, D]

                    outputs = model(sub_tube, obj_tube)
                    logits = outputs["logits"].squeeze(0)   # [C]
                    probs = torch.softmax(logits, dim=-1)
                    top_score, top_class = probs.max(dim=-1)

                    pred = PredictedTriplet(
                        subject_cat=sample.entity_labels[i].item(),
                        object_cat=sample.entity_labels[j].item(),
                        relation_cat=top_class.item(),
                        score=top_score.item(),
                        subject_tube=sample.mask_tubes[i].cpu().numpy() if sample.mask_tubes is not None else None,
                        object_tube=sample.mask_tubes[j].cpu().numpy() if sample.mask_tubes is not None else None,
                    )
                    preds.append(pred)

            all_predictions.append(preds)
            all_ground_truths.append(gts)

    # Compute metrics at each vIoU threshold
    results = []
    for viou_thresh in viou_thresholds:
        result = evaluate_full(all_predictions, all_ground_truths, viou_thresh)
        results.append(result)

    return results


def print_results(results: list[EvaluationResult], title: str = "Evaluation Results"):
    """Pretty-print evaluation results using rich."""
    console = Console()
    table = Table(title=title)

    table.add_column("vIoU Threshold", justify="center")
    table.add_column("R@20", justify="center")
    table.add_column("R@50", justify="center")
    table.add_column("R@100", justify="center")
    table.add_column("mR@20", justify="center")
    table.add_column("mR@50", justify="center")
    table.add_column("mR@100", justify="center")

    for r in results:
        table.add_row(
            f"{r.viou_threshold}",
            f"{r.r_at_20:.4f}",
            f"{r.r_at_50:.4f}",
            f"{r.r_at_100:.4f}",
            f"{r.mr_at_20:.4f}",
            f"{r.mr_at_50:.4f}",
            f"{r.mr_at_100:.4f}",
        )

    console.print(table)
