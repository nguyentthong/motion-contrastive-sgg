"""OpenPVSG dataset loader.

OpenPVSG consists of ~400 videos from ViDOR, Epic-Kitchens, and Ego4D,
with panoptic scene graph annotations including mask tubes and relations.

Expected directory structure after preparation:
    data/openpvsg/
    ├── annotations/
    │   ├── pvsg.json           # Main annotation file
    │   ├── train.json
    │   └── val.json
    ├── videos/                 # Raw video files
    ├── frames/                 # Extracted frames
    │   └── {video_id}/
    │       ├── 000000.jpg
    │       └── ...
    ├── masks/                  # Panoptic segmentation masks
    │   └── {video_id}/
    │       ├── 000000.png
    │       └── ...
    └── precomputed_features/   # Precomputed tube features (optional)
        └── {video_id}.pt

ASSUMPTION: This loader expects precomputed tube features or raw annotations.
Real dataset requires downloading from the OpenPVSG project page.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_aware_tpsgg.datasets.base import (
    TPSGGSample,
    TPSGGDataset,
    RelationAnnotation,
)


# OpenPVSG category definitions (subset for reference)
OPENPVSG_RELATION_CLASSES = [
    "looking at", "not looking at", "unsure", "above", "beneath",
    "in front of", "behind", "on the side of", "in", "on",
    "attached to", "hanging from", "over", "near", "next to",
    "holding", "wearing", "carrying", "standing on", "sitting on",
    "lying on", "leaning on", "playing with", "talking to", "listening to",
    "touching", "reaching for", "throwing", "catching", "kicking",
    "hitting", "biting", "chasing", "running on", "riding",
    "pulling", "pushing", "feeding", "hugging", "kissing",
    "opening", "closing", "drinking from", "eating", "cutting",
    "driving", "cooking", "cleaning", "wiping", "writing on",
    "reading", "texting on", "calling with", "typing on",
    "turning on", "turning off", "pouring into",
]


class OpenPVSGDataset(TPSGGDataset):
    """Dataset loader for OpenPVSG.

    Can load from:
    1. Precomputed features (recommended for relation training)
    2. Raw annotations + frames (requires more processing)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        feature_dim: int = 256,
        num_frames: int = 16,
        use_precomputed: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.use_precomputed = use_precomputed

        self.relation_classes = OPENPVSG_RELATION_CLASSES
        self.num_relation_classes = len(self.relation_classes)

        samples = self._load_samples()
        super().__init__(samples)

    def _load_samples(self) -> list[TPSGGSample]:
        """Load dataset samples from disk."""
        ann_path = self.root_dir / "annotations" / f"{self.split}.json"
        precomputed_dir = self.root_dir / "precomputed_features"

        if not ann_path.exists():
            print(f"[OpenPVSG] Annotation file not found: {ann_path}")
            print("[OpenPVSG] Run scripts/prepare_openpvsg.py to set up the dataset.")
            return []

        with open(ann_path) as f:
            annotations = json.load(f)

        samples = []
        for video_ann in annotations.get("data", []):
            video_id = video_ann["video_id"]

            # Try loading precomputed features
            feat_path = precomputed_dir / f"{video_id}.pt"
            if self.use_precomputed and feat_path.exists():
                data = torch.load(feat_path, map_location="cpu", weights_only=True)
                sample = TPSGGSample(
                    video_id=video_id,
                    num_frames=data.get("num_frames", self.num_frames),
                    entity_labels=data["entity_labels"],
                    tube_features=data["tube_features"],
                    relations=[
                        RelationAnnotation(**r) for r in data["relations"]
                    ],
                )
                samples.append(sample)
            else:
                # Load from raw annotations
                sample = self._load_from_annotations(video_ann)
                if sample is not None:
                    samples.append(sample)

        return samples

    def _load_from_annotations(self, video_ann: dict) -> TPSGGSample | None:
        """Load a sample from raw annotations.

        ASSUMPTION: Without a segmentation model, we create placeholder features.
        In practice, run the segmentation model first and save precomputed features.
        """
        video_id = video_ann["video_id"]
        objects = video_ann.get("objects", {})
        relations = video_ann.get("relations", [])

        if not objects or not relations:
            return None

        entity_ids = list(objects.keys())
        n_entities = len(entity_ids)
        entity_id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # Entity labels
        entity_labels = torch.zeros(n_entities, dtype=torch.long)
        for eid, info in objects.items():
            idx = entity_id_to_idx[eid]
            entity_labels[idx] = info.get("category_id", 0)

        # Placeholder features (should be replaced with real features)
        tube_features = torch.randn(n_entities, self.num_frames, self.feature_dim)

        # Relations
        rel_annotations = []
        for rel in relations:
            sub_id = str(rel.get("subject_id", ""))
            obj_id = str(rel.get("object_id", ""))
            if sub_id in entity_id_to_idx and obj_id in entity_id_to_idx:
                rel_annotations.append(RelationAnnotation(
                    subject_idx=entity_id_to_idx[sub_id],
                    object_idx=entity_id_to_idx[obj_id],
                    relation_cat=rel.get("predicate", 0),
                    start_frame=rel.get("begin_fid", 0),
                    end_frame=rel.get("end_fid", -1),
                ))

        if not rel_annotations:
            return None

        return TPSGGSample(
            video_id=video_id,
            num_frames=self.num_frames,
            entity_labels=entity_labels,
            tube_features=tube_features,
            relations=rel_annotations,
        )
