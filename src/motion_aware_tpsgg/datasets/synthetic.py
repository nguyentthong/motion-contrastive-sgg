"""Synthetic debug dataset for unit testing and example training.

Generates small synthetic videos with random mask tubes, entity labels,
and relation annotations. Allows the full pipeline to run end-to-end.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from motion_aware_tpsgg.datasets.base import TPSGGSample, TPSGGDataset, RelationAnnotation


class SyntheticDebugDataset(TPSGGDataset):
    """Tiny synthetic dataset for testing.

    Generates `num_samples` random video samples, each with random entities
    and relations.
    """

    def __init__(
        self,
        num_samples: int = 50,
        num_frames: int = 8,
        feature_dim: int = 256,
        num_entity_classes: int = 20,
        num_relation_classes: int = 10,
        min_entities: int = 2,
        max_entities: int = 6,
        min_relations: int = 1,
        max_relations: int = 4,
        height: int = 64,
        width: int = 64,
        include_masks: bool = False,
        seed: int = 42,
    ):
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.num_entity_classes = num_entity_classes
        self.num_relation_classes = num_relation_classes
        self.height = height
        self.width = width

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        samples = []
        for i in range(num_samples):
            n_entities = rng.randint(min_entities, max_entities)
            n_relations = min(rng.randint(min_relations, max_relations),
                              n_entities * (n_entities - 1))

            entity_labels = torch.randint(0, num_entity_classes, (n_entities,))
            tube_features = torch.randn(n_entities, num_frames, feature_dim)

            # Generate random relations
            all_pairs = [(s, o) for s in range(n_entities)
                         for o in range(n_entities) if s != o]
            rng.shuffle(all_pairs)
            selected_pairs = all_pairs[:n_relations]

            relations = []
            for sub_idx, obj_idx in selected_pairs:
                rel_cat = rng.randint(0, num_relation_classes - 1)
                start = rng.randint(0, max(0, num_frames - 2))
                end = rng.randint(start + 1, num_frames)
                relations.append(RelationAnnotation(
                    subject_idx=sub_idx,
                    object_idx=obj_idx,
                    relation_cat=rel_cat,
                    start_frame=start,
                    end_frame=end,
                ))

            # Optional mask tubes
            mask_tubes = None
            if include_masks:
                mask_tubes = torch.zeros(n_entities, num_frames, height, width)
                for e in range(n_entities):
                    y0 = (e * height // n_entities)
                    y1 = min(y0 + height // n_entities, height)
                    x0 = rng.randint(0, width // 2)
                    x1 = x0 + width // 3
                    mask_tubes[e, :, y0:y1, x0:x1] = 1.0

            sample = TPSGGSample(
                video_id=f"synthetic_{i:04d}",
                num_frames=num_frames,
                entity_labels=entity_labels,
                tube_features=tube_features,
                relations=relations,
                mask_tubes=mask_tubes,
            )
            samples.append(sample)

        super().__init__(samples)

    @staticmethod
    def get_category_names(num_classes: int, prefix: str = "cls") -> list[str]:
        """Generate dummy category names."""
        return [f"{prefix}_{i}" for i in range(num_classes)]
