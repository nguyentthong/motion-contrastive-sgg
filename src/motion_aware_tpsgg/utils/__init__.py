"""Utility functions for config loading, seeding, and common operations."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str) -> None:
    """Save config to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def build_from_config(config: dict[str, Any]):
    """Build model, loss, and training configs from a flat YAML config dict.

    Returns:
        Tuple of (ModelConfig, LossConfig, TrainConfig).
    """
    from motion_aware_tpsgg.models import ModelConfig
    from motion_aware_tpsgg.losses import LossConfig
    from motion_aware_tpsgg.training import TrainConfig

    model_cfg = ModelConfig(
        input_dim=config.get("input_dim", 256),
        hidden_dim=config.get("hidden_dim", 256),
        num_relation_classes=config.get("num_relation_classes", 57),
        num_entity_classes=config.get("num_entity_classes", 133),
        tube_encoder_type=config.get("tube_encoder_type", "convolution"),
        num_encoder_layers=config.get("num_encoder_layers", 2),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
        max_tube_length=config.get("max_tube_length", 64),
    )

    loss_cfg = LossConfig(
        weight_relation_ce=config.get("weight_relation_ce", 1.0),
        weight_shuffle_contrastive=config.get("weight_shuffle_contrastive", 1.0),
        weight_triplet_contrastive=config.get("weight_triplet_contrastive", 1.0),
        similarity_method=config.get("similarity_method", "optimal_transport"),
        ot_alpha=config.get("alpha", 10.0),
        ot_n_iter=config.get("sinkhorn_iterations", 1000),
        ot_tau=config.get("sinkhorn_tau", 0.05),
        gamma=config.get("gamma", 9.0),
    )

    train_cfg = TrainConfig(
        optimizer=config.get("optimizer", "adam"),
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
        gradient_clip_norm=config.get("gradient_clip_norm", 0.0),
        num_epochs=config.get("num_epochs", 50),
        batch_size=config.get("batch_size", 32),
        num_frames=config.get("num_frames", 8),
        pad_length=config.get("pad_length", 16),
        use_contrastive=config.get("use_contrastive", True),
        save_dir=config.get("save_dir", "checkpoints"),
        save_every=config.get("save_every", 5),
        seed=config.get("seed", 42),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model_cfg, loss_cfg, train_cfg
