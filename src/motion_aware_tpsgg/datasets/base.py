"""Base dataset class and data structures for TPSGG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RelationAnnotation:
    """A single relation annotation: subject-relation-object with temporal span."""
    subject_idx: int          # Index into entity list
    object_idx: int           # Index into entity list
    relation_cat: int         # Relation category index
    start_frame: int = 0
    end_frame: int = -1       # -1 means entire clip


@dataclass
class TPSGGSample:
    """A single video sample with all annotations for TPSGG.

    Contains precomputed tube features for efficient relation training.
    """
    video_id: str
    num_frames: int

    # Entity information
    entity_labels: torch.Tensor           # [N] category indices
    tube_features: torch.Tensor           # [N, T, D] precomputed tube features

    # Relation annotations
    relations: list[RelationAnnotation]

    # Optional raw data
    mask_tubes: torch.Tensor | None = None           # [N, T, H, W] binary
    frames: torch.Tensor | None = None               # [T, C, H, W]
    flow_magnitude: np.ndarray | None = None         # [T-1, H, W]

    # Metadata
    num_entities: int = 0
    feature_dim: int = 256

    def __post_init__(self):
        self.num_entities = self.entity_labels.shape[0]
        if self.tube_features.dim() == 3:
            self.feature_dim = self.tube_features.shape[2]


@dataclass
class RelationBatch:
    """A collated batch of subject-object pairs for relation prediction."""
    subject_tubes: torch.Tensor      # [B, T, D] subject tube features
    object_tubes: torch.Tensor       # [B, T, D] object tube features
    relation_labels: torch.Tensor    # [B] relation class labels
    subject_cats: torch.Tensor       # [B] subject category labels
    object_cats: torch.Tensor        # [B] object category labels
    video_ids: list[str]             # [B] video identifiers
    pair_indices: list[tuple[int, int]]  # [B] (subject_idx, object_idx)


class TPSGGDataset(Dataset):
    """Base dataset that yields TPSGGSample objects."""

    def __init__(self, samples: list[TPSGGSample] | None = None):
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TPSGGSample:
        return self.samples[idx]


def collate_relation_pairs(
    samples: list[TPSGGSample],
    max_pairs_per_sample: int = 50,
    pad_length: int | None = None,
) -> RelationBatch:
    """Collate samples into a batch of subject-object relation pairs.

    For each sample, creates all annotated relation pairs.

    Args:
        samples: List of TPSGGSample.
        max_pairs_per_sample: Max pairs to extract per sample.
        pad_length: If set, pad temporal dimension to this length.

    Returns:
        RelationBatch ready for model forward pass.
    """
    all_sub_tubes = []
    all_obj_tubes = []
    all_rel_labels = []
    all_sub_cats = []
    all_obj_cats = []
    all_video_ids = []
    all_pair_indices = []

    for sample in samples:
        count = 0
        for rel in sample.relations:
            if count >= max_pairs_per_sample:
                break
            sub_idx = rel.subject_idx
            obj_idx = rel.object_idx

            if sub_idx >= sample.num_entities or obj_idx >= sample.num_entities:
                continue

            sub_tube = sample.tube_features[sub_idx]  # [T, D]
            obj_tube = sample.tube_features[obj_idx]  # [T, D]

            all_sub_tubes.append(sub_tube)
            all_obj_tubes.append(obj_tube)
            all_rel_labels.append(rel.relation_cat)
            all_sub_cats.append(sample.entity_labels[sub_idx].item())
            all_obj_cats.append(sample.entity_labels[obj_idx].item())
            all_video_ids.append(sample.video_id)
            all_pair_indices.append((sub_idx, obj_idx))
            count += 1

    if not all_sub_tubes:
        # Return empty batch
        D = samples[0].feature_dim if samples else 256
        T = pad_length or 8
        return RelationBatch(
            subject_tubes=torch.zeros(0, T, D),
            object_tubes=torch.zeros(0, T, D),
            relation_labels=torch.zeros(0, dtype=torch.long),
            subject_cats=torch.zeros(0, dtype=torch.long),
            object_cats=torch.zeros(0, dtype=torch.long),
            video_ids=[],
            pair_indices=[],
        )

    # Pad temporal dimension
    if pad_length is not None:
        padded_sub = []
        padded_obj = []
        for s, o in zip(all_sub_tubes, all_obj_tubes):
            T, D = s.shape
            if T < pad_length:
                pad_s = torch.zeros(pad_length - T, D, device=s.device)
                s = torch.cat([s, pad_s], dim=0)
                o = torch.cat([o, pad_s], dim=0)
            else:
                s = s[:pad_length]
                o = o[:pad_length]
            padded_sub.append(s)
            padded_obj.append(o)
        all_sub_tubes = padded_sub
        all_obj_tubes = padded_obj

    return RelationBatch(
        subject_tubes=torch.stack(all_sub_tubes),
        object_tubes=torch.stack(all_obj_tubes),
        relation_labels=torch.tensor(all_rel_labels, dtype=torch.long),
        subject_cats=torch.tensor(all_sub_cats, dtype=torch.long),
        object_cats=torch.tensor(all_obj_cats, dtype=torch.long),
        video_ids=all_video_ids,
        pair_indices=all_pair_indices,
    )
