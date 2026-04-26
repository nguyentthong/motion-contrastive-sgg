"""Dataset modules for temporal panoptic scene graph generation.

Supports:
    - OpenPVSG: natural videos from ViDOR, Epic-Kitchens, Ego4D
    - PSG4D: 4D scene graphs (PSG4D-GTA, PSG4D-HOI) with RGB-D and point clouds
    - SyntheticDebug: tiny synthetic dataset for testing
"""

from motion_aware_tpsgg.datasets.base import TPSGGSample, TPSGGDataset
from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
from motion_aware_tpsgg.datasets.openpvsg import OpenPVSGDataset
from motion_aware_tpsgg.datasets.psg4d import PSG4DDataset

__all__ = [
    "TPSGGSample",
    "TPSGGDataset",
    "SyntheticDebugDataset",
    "OpenPVSGDataset",
    "PSG4DDataset",
]
