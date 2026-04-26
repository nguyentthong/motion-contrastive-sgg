#!/usr/bin/env python3
"""Evaluation script for temporal panoptic scene graph generation.

Usage:
    uv run python scripts/evaluate.py --config configs/synthetic.yaml --checkpoint checkpoints/synthetic/checkpoint_final.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from motion_aware_tpsgg.utils import load_config, build_from_config, set_seed
from motion_aware_tpsgg.models import TPSGGModel
from motion_aware_tpsgg.evaluation import run_evaluation, print_results
from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset
from motion_aware_tpsgg.datasets.openpvsg import OpenPVSGDataset
from motion_aware_tpsgg.datasets.psg4d import PSG4DDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate TPSGG model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg, loss_cfg, train_cfg = build_from_config(config)
    set_seed(train_cfg.seed)

    # Build dataset
    dataset_type = config.get("dataset", "synthetic")
    if dataset_type == "synthetic":
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
        val_ds = OpenPVSGDataset(
            config.get("data_root", "data/openpvsg"),
            split=args.split,
            feature_dim=config.get("input_dim", 256),
        )
    elif dataset_type == "psg4d":
        val_ds = PSG4DDataset(
            config.get("data_root", "data/psg4d/psg4d_gta"),
            split=args.split,
            modality=config.get("modality", "rgbd"),
            feature_dim=config.get("input_dim", 256),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    print(f"Evaluation samples: {len(val_ds)}")
    if len(val_ds) == 0:
        print("No samples to evaluate.")
        return

    # Build model
    model = TPSGGModel(model_cfg)

    # Load checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint provided. Evaluating with random weights.")

    # Run evaluation
    device = config.get("device", "cpu")
    results = run_evaluation(
        model, val_ds, device=device,
        pad_length=config.get("pad_length", 16),
    )

    print_results(results, title=f"Results on {dataset_type}")


if __name__ == "__main__":
    main()
