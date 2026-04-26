"""Segmentation backbone abstractions for Stage A of the pipeline.

Provides clean interfaces for:
    - IPS+T: Image Panoptic Segmentation + Tracker (Mask2Former + UniTrack)
    - VPS: Video Panoptic Segmentation (Video K-Net)
    - PSG4DFormer: 4D segmentation for RGB-D / point cloud

These are wrapper classes. Actual heavyweight models (Mask2Former, Video K-Net, etc.)
require separate installation and are behind optional imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SegmentationOutput:
    """Output from a panoptic segmentation module."""
    mask_tubes: torch.Tensor          # [N, T, H, W] binary mask tubes
    entity_labels: torch.Tensor       # [N] predicted entity class indices
    entity_scores: torch.Tensor       # [N] confidence scores
    tube_features: torch.Tensor       # [N, T, D] per-entity tube features for relation model
    num_entities: int


class SegmentationBackbone(ABC):
    """Abstract interface for temporal panoptic segmentation (Stage A)."""

    @abstractmethod
    def segment(self, video: torch.Tensor) -> SegmentationOutput:
        """Segment a video into mask tubes.

        Args:
            video: Input video tensor. Shape depends on modality.

        Returns:
            SegmentationOutput with mask tubes and features.
        """
        ...


class PrecomputedSegmentation(SegmentationBackbone):
    """Load precomputed segmentation outputs.

    This is the recommended approach for training the relation model:
    run segmentation offline, save results, and load them here.
    """

    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim

    def segment(self, video: torch.Tensor) -> SegmentationOutput:
        raise NotImplementedError(
            "PrecomputedSegmentation should load from disk, not run on raw video. "
            "Use load_precomputed() instead."
        )

    def load_precomputed(
        self,
        mask_tubes: torch.Tensor,
        entity_labels: torch.Tensor,
        tube_features: torch.Tensor,
        entity_scores: torch.Tensor | None = None,
    ) -> SegmentationOutput:
        """Wrap precomputed data into SegmentationOutput."""
        N = mask_tubes.shape[0]
        if entity_scores is None:
            entity_scores = torch.ones(N)
        return SegmentationOutput(
            mask_tubes=mask_tubes,
            entity_labels=entity_labels,
            entity_scores=entity_scores,
            tube_features=tube_features,
            num_entities=N,
        )


class DummySegmentation(SegmentationBackbone):
    """Dummy segmentation for testing/debugging."""

    def __init__(
        self,
        num_entities: int = 5,
        num_frames: int = 8,
        height: int = 64,
        width: int = 64,
        feature_dim: int = 256,
        num_classes: int = 133,
    ):
        self.num_entities = num_entities
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def segment(self, video: torch.Tensor) -> SegmentationOutput:
        device = video.device
        N = self.num_entities
        T = self.num_frames

        mask_tubes = torch.zeros(N, T, self.height, self.width, device=device)
        for i in range(N):
            y0 = (i * self.height // N)
            y1 = min(y0 + self.height // N, self.height)
            mask_tubes[i, :, y0:y1, :] = 1.0

        entity_labels = torch.randint(0, self.num_classes, (N,), device=device)
        entity_scores = torch.ones(N, device=device)
        tube_features = torch.randn(N, T, self.feature_dim, device=device)

        return SegmentationOutput(
            mask_tubes=mask_tubes,
            entity_labels=entity_labels,
            entity_scores=entity_scores,
            tube_features=tube_features,
            num_entities=N,
        )


# ==================== External model wrappers (optional) ====================

class Mask2FormerIPST(SegmentationBackbone):
    """Wrapper for Mask2Former + UniTrack (IPS+T pipeline).

    Requires: detectron2, mask2former packages.
    """

    def __init__(self, config_path: str | None = None, weights_path: str | None = None):
        self.config_path = config_path
        self.weights_path = weights_path
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            # This would be the actual Mask2Former + tracker setup
            raise ImportError("Mask2Former integration requires detectron2. "
                              "Install with: pip install detectron2")
        except ImportError as e:
            raise ImportError(
                f"Mask2Former not available: {e}\n"
                "Use PrecomputedSegmentation or DummySegmentation instead."
            ) from e

    def segment(self, video: torch.Tensor) -> SegmentationOutput:
        self._lazy_init()
        raise NotImplementedError("Full Mask2Former integration pending.")


class VideoKNetVPS(SegmentationBackbone):
    """Wrapper for Video K-Net (VPS pipeline).

    Requires: mmdet, mmcv packages.
    """

    def __init__(self, config_path: str | None = None, weights_path: str | None = None):
        self.config_path = config_path
        self.weights_path = weights_path

    def segment(self, video: torch.Tensor) -> SegmentationOutput:
        raise NotImplementedError(
            "Video K-Net integration requires mmdet/mmcv. "
            "Use PrecomputedSegmentation or DummySegmentation instead."
        )


class PSG4DFormerBackbone(SegmentationBackbone):
    """Wrapper for PSG4DFormer (4D segmentation for RGB-D / point cloud).

    Requires: specialized 4D segmentation libraries.
    """

    def __init__(self, modality: str = "rgbd"):
        self.modality = modality  # 'rgbd' or 'pointcloud'

    def segment(self, video: torch.Tensor) -> SegmentationOutput:
        raise NotImplementedError(
            "PSG4DFormer integration requires specialized libraries. "
            "Use PrecomputedSegmentation or DummySegmentation instead."
        )
