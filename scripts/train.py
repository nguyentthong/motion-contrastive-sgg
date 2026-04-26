#!/usr/bin/env python3
"""Training script for temporal panoptic scene graph generation.

Usage:
    uv run python scripts/train.py --config configs/synthetic.yaml
    uv run python scripts/train.py --config configs/openpvsg.yaml
    uv run python scripts/train.py --config configs/psg4d_rgbd.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motion_aware_tpsgg.utils import load_config, build_from_config, set_seed
from motion_aware_tpsgg.models import TPSGGModel
from motion_aware_tpsgg.training import train
from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
from motion_aware_tpsgg.datasets.openpvsg import OpenPVSGDataset
from motion_aware_tpsgg.datasets.psg4d import PSG4DDataset


def build_datasets(config: dict):
    """Build train/val datasets from config."""
    dataset_type = config.get("dataset", "synthetic")

    if dataset_type == "synthetic":
        train_ds = SyntheticDebugDataset(
            num_samples=config.get("num_samples_train", 50),
            num_frames=config.get("num_frames", 8),
            feature_dim=config.get("input_dim", 256),
            num_entity_classes=config.get("num_entity_classes", 20),
            num_relation_classes=config.get("num_relation_classes", 10),
            include_masks=True,
            seed=config.get("seed", 42),
        )
        val_ds = SyntheticDebugDataset(
            num_samples=config.get("num_samples_val", 10),
            num_frames=config.get("num_frames", 8),
            feature_dim=config.get("input_dim", 256),
            num_entity_classes=config.get("num_entity_classes", 20),
            num_relation_classes=config.get("num_relation_classes", 10),
            include_masks=True,
            seed=config.get("seed", 42) + 1000,
        )
    elif dataset_type == "openpvsg":
        root = config.get("data_root", "data/openpvsg")
        train_ds = OpenPVSGDataset(root, split="train",
                                    feature_dim=config.get("input_dim", 256),
                                    num_frames=config.get("num_frames", 16))
        val_ds = OpenPVSGDataset(root, split="val",
                                  feature_dim=config.get("input_dim", 256),
                                  num_frames=config.get("num_frames", 16))
    elif dataset_type == "psg4d":
        root = config.get("data_root", "data/psg4d/psg4d_gta")
        modality = config.get("modality", "rgbd")
        train_ds = PSG4DDataset(root, split="train", modality=modality,
                                 feature_dim=config.get("input_dim", 256),
                                 num_frames=config.get("num_frames", 16))
        val_ds = PSG4DDataset(root, split="val", modality=modality,
                               feature_dim=config.get("input_dim", 256),
                               num_frames=config.get("num_frames", 16))
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train TPSGG model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg, loss_cfg, train_cfg = build_from_config(config)

    set_seed(train_cfg.seed)
    print(f"Config: {args.config}")
    print(f"Encoder: {model_cfg.tube_encoder_type}, Contrastive: {train_cfg.use_contrastive}")
    print(f"Similarity: {loss_cfg.similarity_method}, Device: {train_cfg.device}")

    train_ds, val_ds = build_datasets(config)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training samples found. Check data paths and preparation scripts.")
        sys.exit(1)

    model = TPSGGModel(model_cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    trained_model = train(model, train_ds, val_ds, loss_cfg, train_cfg)
    print("Done.")


if __name__ == "__main__":
    main()
