"""PSG4D dataset loader for 4D panoptic scene graph generation.

Supports PSG4D-GTA and PSG4D-HOI splits with RGB-D and point cloud modalities.

Expected directory structure:
    data/psg4d/
    ├── psg4d_gta/
    │   ├── annotations/
    │   │   ├── train.json
    │   │   └── val.json
    │   ├── rgb/            # RGB frames
    │   ├── depth/          # Depth maps
    │   ├── pointclouds/    # Point cloud files (.ply)
    │   └── precomputed_features/
    └── psg4d_hoi/
        ├── annotations/
        ├── rgb/
        ├── depth/
        ├── pointclouds/
        └── precomputed_features/

PSG4D-GTA: 67 third-view videos, 35 object categories, 43 relation categories
PSG4D-HOI: 2973 egocentric videos, 46 object categories, 15 relation categories
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from motion_aware_tpsgg.datasets.base import (
    TPSGGSample,
    TPSGGDataset,
    RelationAnnotation,
)


class PSG4DDataset(TPSGGDataset):
    """Dataset loader for PSG4D (GTA and HOI splits).

    Args:
        root_dir: Path to psg4d_gta/ or psg4d_hoi/.
        split: 'train' or 'val'.
        modality: 'rgbd' or 'pointcloud'.
        feature_dim: Dimension of tube features.
        num_frames: Number of frames per clip.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        modality: str = "rgbd",
        feature_dim: int = 256,
        num_frames: int = 16,
        use_precomputed: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.modality = modality
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.use_precomputed = use_precomputed

        samples = self._load_samples()
        super().__init__(samples)

    def _load_samples(self) -> list[TPSGGSample]:
        ann_path = self.root_dir / "annotations" / f"{self.split}.json"
        precomputed_dir = self.root_dir / "precomputed_features" / self.modality

        if not ann_path.exists():
            print(f"[PSG4D] Annotation file not found: {ann_path}")
            print("[PSG4D] Run scripts/prepare_psg4d.py to set up the dataset.")
            return []

        with open(ann_path) as f:
            annotations = json.load(f)

        samples = []
        for video_ann in annotations.get("data", []):
            video_id = video_ann["video_id"]

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
                sample = self._load_from_annotations(video_ann)
                if sample is not None:
                    samples.append(sample)

        return samples

    def _load_from_annotations(self, video_ann: dict) -> TPSGGSample | None:
        """Load from raw annotations with placeholder features."""
        video_id = video_ann["video_id"]
        objects = video_ann.get("objects", {})
        relations = video_ann.get("relations", [])

        if not objects or not relations:
            return None

        entity_ids = list(objects.keys())
        n_entities = len(entity_ids)
        entity_id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        entity_labels = torch.zeros(n_entities, dtype=torch.long)
        for eid, info in objects.items():
            idx = entity_id_to_idx[eid]
            entity_labels[idx] = info.get("category_id", 0)

        tube_features = torch.randn(n_entities, self.num_frames, self.feature_dim)

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
