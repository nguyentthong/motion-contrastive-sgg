#!/usr/bin/env python3
"""Quick synthetic debug script - trains and evaluates on synthetic data.

Usage:
    uv run python scripts/run_synthetic_debug.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motion_aware_tpsgg.utils import set_seed
from motion_aware_tpsgg.models import TPSGGModel, ModelConfig
from motion_aware_tpsgg.losses import LossConfig
from motion_aware_tpsgg.training import TrainConfig, train
from motion_aware_tpsgg.evaluation import run_evaluation, print_results
from motion_aware_tpsgg.datasets.synthetic import SyntheticDebugDataset


def main():
    print("=" * 60)
    print("Motion-aware TPSGG - Synthetic Debug Run")
    print("=" * 60)

    set_seed(42)

    # Small configs for quick testing
    model_cfg = ModelConfig(
        input_dim=64,
        hidden_dim=64,
        num_relation_classes=10,
        num_entity_classes=20,
        tube_encoder_type="convolution",
        num_encoder_layers=1,
        max_tube_length=16,
    )

    loss_cfg = LossConfig(
        similarity_method="pooling_cosine",  # Fast for debug
        gamma=0.0,  # Apply shuffle to all tubes
    )

    train_cfg = TrainConfig(
        num_epochs=3,
        batch_size=8,
        learning_rate=1e-3,
        pad_length=16,
        use_contrastive=True,
        save_dir="checkpoints/debug",
        save_every=2,
        device="cpu",
    )

    # Datasets
    train_ds = SyntheticDebugDataset(
        num_samples=30, num_frames=8, feature_dim=64,
        num_entity_classes=20, num_relation_classes=10,
        include_masks=True, seed=42,
    )
    val_ds = SyntheticDebugDataset(
        num_samples=10, num_frames=8, feature_dim=64,
        num_entity_classes=20, num_relation_classes=10,
        include_masks=True, seed=123,
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Build and train
    model = TPSGGModel(model_cfg)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n--- Training ---")
    model = train(model, train_ds, val_ds, loss_cfg, train_cfg)

    # Evaluate
    print("\n--- Evaluation ---")
    results = run_evaluation(model, val_ds, device="cpu", pad_length=16)
    print_results(results, title="Synthetic Debug Results")

    print("\nSynthetic debug run complete!")


if __name__ == "__main__":
    main()
