"""Training loop for relation classifier with motion-aware contrastive learning.

Supports:
    A) Relation classifier only (cross-entropy)
    B) Relation classifier + contrastive learning (CE + shuffle + triplet)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from motion_aware_tpsgg.datasets.base import (
    TPSGGDataset,
    TPSGGSample,
    RelationBatch,
    collate_relation_pairs,
)
from motion_aware_tpsgg.losses import LossConfig, MotionAwareContrastiveLoss
from motion_aware_tpsgg.models import TPSGGModel, ModelConfig
from motion_aware_tpsgg.sampling import (
    Triplet,
    PositiveSampler,
    ShuffleNegativeSampler,
    TripletNegativeSampler,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = "adam"              # 'adam' or 'adamw'
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 0.0     # 0 = no clipping

    # Schedule
    num_epochs: int = 50
    batch_size: int = 32

    # Data
    num_frames: int = 8
    pad_length: int = 16
    max_pairs_per_sample: int = 50

    # Contrastive learning
    use_contrastive: bool = True
    num_shuffle_negatives: int = 3
    num_triplet_negatives: int = 3

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 5
    log_every: int = 10

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _build_triplets_for_contrastive(
    samples: list[TPSGGSample],
    model: TPSGGModel,
    device: torch.device,
) -> list[Triplet]:
    """Build Triplet objects from dataset samples for contrastive sampling."""
    triplets = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            if sample.num_entities < 2:
                continue
            tube_feats = sample.tube_features.to(device)  # [N, T, D]
            H = model.encode_tubes(tube_feats)             # [N, T, D]

            for rel in sample.relations:
                if rel.subject_idx >= sample.num_entities or rel.object_idx >= sample.num_entities:
                    continue
                t = Triplet(
                    video_id=sample.video_id,
                    subject_cat=sample.entity_labels[rel.subject_idx].item(),
                    relation_cat=rel.relation_cat,
                    object_cat=sample.entity_labels[rel.object_idx].item(),
                    subject_tube=H[rel.subject_idx],
                    object_tube=H[rel.object_idx],
                )
                t.build_anchor()
                triplets.append(t)
    model.train()
    return triplets


def train_one_epoch(
    model: TPSGGModel,
    dataloader: DataLoader,
    loss_fn: MotionAwareContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    positive_sampler: PositiveSampler | None = None,
    shuffle_sampler: ShuffleNegativeSampler | None = None,
    triplet_sampler: TripletNegativeSampler | None = None,
    all_triplets: list[Triplet] | None = None,
    video_triplet_index: dict[str, list[Triplet]] | None = None,
) -> dict[str, float]:
    """Train for one epoch.

    Returns dict of average losses.
    """
    model.train()
    device = torch.device(config.device)
    total_losses = {"total": 0.0, "relation_ce": 0.0, "shuffle": 0.0, "triplet": 0.0}
    num_batches = 0

    for batch_idx, batch_samples in enumerate(dataloader):
        if isinstance(batch_samples, TPSGGSample):
            batch_samples = [batch_samples]
        elif isinstance(batch_samples, list):
            pass
        else:
            batch_samples = [batch_samples]

        # Collate into relation pairs
        batch = collate_relation_pairs(
            batch_samples,
            max_pairs_per_sample=config.max_pairs_per_sample,
            pad_length=config.pad_length,
        )

        if batch.subject_tubes.shape[0] == 0:
            continue

        sub_tubes = batch.subject_tubes.to(device)
        obj_tubes = batch.object_tubes.to(device)
        rel_labels = batch.relation_labels.to(device)

        # Forward pass
        outputs = model(sub_tubes, obj_tubes)
        logits = outputs["logits"]

        # Build contrastive data
        shuffle_data = None
        triplet_data = None

        if config.use_contrastive and all_triplets and positive_sampler:
            # Build contrastive pairs from current batch
            anchors_s, positives_s, negatives_s = [], [], []
            anchors_t, positives_t, negatives_t = [], [], []

            H_sub = outputs["H_sub"]  # [B, T, D]
            H_obj = outputs["H_obj"]  # [B, T, D]

            for i in range(H_sub.shape[0]):
                # Build anchor [T, 2D]
                anchor_repr = torch.cat([H_sub[i], H_obj[i]], dim=-1)

                # Build a dummy Triplet for sampling
                anchor_triplet = Triplet(
                    video_id=batch.video_ids[i],
                    subject_cat=batch.subject_cats[i].item(),
                    relation_cat=batch.relation_labels[i].item(),
                    object_cat=batch.object_cats[i].item(),
                    subject_tube=H_sub[i],
                    object_tube=H_obj[i],
                    anchor_repr=anchor_repr,
                )

                # Positive sampling
                pos = positive_sampler.sample(anchor_triplet)
                if pos is None:
                    continue

                pos_repr = pos.anchor_repr
                if pos_repr is None:
                    pos_repr = pos.build_anchor()
                pos_repr = pos_repr.to(device)

                # Shuffle-based negatives
                if shuffle_sampler:
                    shuffle_negs = shuffle_sampler.sample(anchor_triplet)
                    if shuffle_negs:
                        anchors_s.append(anchor_repr)
                        positives_s.append(pos_repr)
                        negatives_s.append(shuffle_negs)

                # Triplet-based negatives
                if triplet_sampler and video_triplet_index:
                    vid = batch.video_ids[i]
                    same_video = video_triplet_index.get(vid, [])
                    triplet_negs = triplet_sampler.sample(anchor_triplet, same_video)
                    if triplet_negs:
                        anchors_t.append(anchor_repr)
                        positives_t.append(pos_repr)
                        neg_on_device = [n.to(device) for n in triplet_negs]
                        negatives_t.append(neg_on_device)

            if anchors_s:
                shuffle_data = {
                    "anchors": anchors_s,
                    "positives": positives_s,
                    "negatives": negatives_s,
                }
            if anchors_t:
                triplet_data = {
                    "anchors": anchors_t,
                    "positives": positives_t,
                    "negatives": negatives_t,
                }

        # Compute loss
        losses = loss_fn(logits, rel_labels, shuffle_data, triplet_data)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        # Accumulate
        total_losses["total"] += losses["total"].item()
        total_losses["relation_ce"] += losses["relation_ce"].item()
        total_losses["shuffle"] += losses["shuffle_contrastive"].item()
        total_losses["triplet"] += losses["triplet_contrastive"].item()
        num_batches += 1

    # Average
    if num_batches > 0:
        for k in total_losses:
            total_losses[k] /= num_batches

    return total_losses


def train(
    model: TPSGGModel,
    train_dataset: TPSGGDataset,
    val_dataset: TPSGGDataset | None,
    loss_config: LossConfig,
    train_config: TrainConfig,
) -> TPSGGModel:
    """Full training loop.

    Args:
        model: The TPSGG model.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        loss_config: Loss configuration.
        train_config: Training configuration.

    Returns:
        Trained model.
    """
    device = torch.device(train_config.device)
    model = model.to(device)

    # Build optimizer
    if train_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
        )

    # Loss
    loss_fn = MotionAwareContrastiveLoss(loss_config, model.config.num_relation_classes)

    # DataLoader (returns list of TPSGGSample)
    def collate_fn(batch):
        return batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Build contrastive samplers
    positive_sampler = None
    shuffle_sampler = None
    triplet_sampler = None
    all_triplets: list[Triplet] = []
    video_triplet_index: dict[str, list[Triplet]] = {}

    if train_config.use_contrastive:
        print("Building contrastive sampling index...")
        all_triplets = _build_triplets_for_contrastive(
            train_dataset.samples, model, device
        )
        positive_sampler = PositiveSampler(all_triplets)
        shuffle_sampler = ShuffleNegativeSampler(
            gamma=loss_config.gamma,
            num_negatives=train_config.num_shuffle_negatives,
        )
        triplet_sampler = TripletNegativeSampler(
            num_negatives=train_config.num_triplet_negatives,
        )
        for t in all_triplets:
            if t.video_id not in video_triplet_index:
                video_triplet_index[t.video_id] = []
            video_triplet_index[t.video_id].append(t)
        print(f"  {len(all_triplets)} triplets from {len(video_triplet_index)} videos")

    # Save dir
    save_dir = Path(train_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, train_config.num_epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, train_config, epoch,
            positive_sampler, shuffle_sampler, triplet_sampler,
            all_triplets, video_triplet_index,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{train_config.num_epochs} "
            f"[{elapsed:.1f}s] "
            f"total={losses['total']:.4f} "
            f"ce={losses['relation_ce']:.4f} "
            f"shuffle={losses['shuffle']:.4f} "
            f"triplet={losses['triplet']:.4f}"
        )

        # Save checkpoint
        if epoch % train_config.save_every == 0:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final
    final_path = save_dir / "checkpoint_final.pt"
    torch.save({
        "epoch": train_config.num_epochs,
        "model_state_dict": model.state_dict(),
    }, final_path)
    print(f"Training complete. Final model: {final_path}")

    return model
