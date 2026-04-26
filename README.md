# Motion-aware Contrastive Learning for Temporal Panoptic Scene Graph Generation

A PyTorch implementation of the paper:

> **Motion-aware Contrastive Learning for Temporal Panoptic Scene Graph Generation**
> Thong Thanh Nguyen, Xiaobao Wu, Yi Bin, Cong-Duy Nguyen, See-Kiong Ng, Anh Tuan Luu
> AAAI 2025 — [arXiv:2412.07160v2](https://arxiv.org/abs/2412.07160v2)

## Overview

This codebase implements the two-stage pipeline for temporal panoptic scene graph generation:

- **Stage A** — Temporal panoptic segmentation (pluggable backbone wrappers for Mask2Former + UniTrack, Video K-Net, PSG4DFormer)
- **Stage B** — Relation classification with motion-aware contrastive learning, including:
  - Shuffle-based contrastive learning (temporally permuted negatives for strong-motion tubes)
  - Triplet-based contrastive learning (hard negatives from same video)
  - Optimal transport distance for mask tube similarity

## Setup

### With `uv` (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Verify installation

```bash
# Run test suite (35 tests)
pytest tests/test_all.py -v

# Quick synthetic debug (trains 3 epochs, evaluates, ~2 seconds)
python scripts/run_synthetic_debug.py
```

## Repository Structure

```
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── synthetic.yaml          # Debug/testing config
│   ├── openpvsg.yaml           # OpenPVSG with paper hyperparams
│   ├── psg4d_rgbd.yaml         # PSG4D RGB-D config
│   └── psg4d_pointcloud.yaml   # PSG4D point cloud config
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── evaluate.py             # Evaluation entry point
│   ├── run_synthetic_debug.py  # Quick end-to-end test
│   ├── download_openpvsg.py    # Dataset prep instructions
│   ├── download_psg4d.py       # Dataset prep instructions
│   ├── prepare_openpvsg.py     # Split annotations
│   └── prepare_psg4d.py        # Split annotations
├── src/motion_aware_tpsgg/
│   ├── datasets/               # Dataset loaders
│   │   ├── base.py             # TPSGGSample, collation
│   │   ├── synthetic.py        # Synthetic debug dataset
│   │   ├── openpvsg.py         # OpenPVSG loader
│   │   └── psg4d.py            # PSG4D loader (GTA/HOI, RGB-D/pointcloud)
│   ├── models/
│   │   ├── __init__.py         # TPSGGModel, tube encoders, relation classifier
│   │   └── segmentation.py     # Backbone wrappers (Stage A)
│   ├── losses/                 # InfoNCE contrastive + cross-entropy
│   ├── metrics/                # R@K, mR@K, vIoU
│   ├── sampling/               # Positive, shuffle-negative, triplet-negative
│   ├── transport/              # Sinkhorn optimal transport distance
│   ├── training/               # Training loop with contrastive integration
│   ├── evaluation/             # Inference + metric computation
│   └── utils/                  # Config loading, seeding
└── tests/
    └── test_all.py             # 35 unit tests
```

## Dataset Preparation

### Synthetic (no download needed)

Used automatically for testing. Configured via `configs/synthetic.yaml`.

### OpenPVSG

```bash
# 1. Create directory structure
python scripts/download_openpvsg.py --output data/openpvsg

# 2. Download data from https://github.com/Jingkang50/OpenPSG
#    Place pvsg.json in data/openpvsg/annotations/

# 3. Generate train/val splits
python scripts/prepare_openpvsg.py --root data/openpvsg
```

**Expected structure:**
```
data/openpvsg/
├── annotations/
│   ├── pvsg.json
│   ├── train.json
│   └── val.json
├── frames/{video_id}/*.jpg
├── masks/{video_id}/*.png
└── precomputed_features/{video_id}.pt   # ← recommended
```

### PSG4D

```bash
# 1. Create directory structure
python scripts/download_psg4d.py --output data/psg4d

# 2. Download from https://github.com/Jingkang50/4D-PSG
# 3. Prepare splits
python scripts/prepare_psg4d.py --root data/psg4d/psg4d_gta
python scripts/prepare_psg4d.py --root data/psg4d/psg4d_hoi
```

### Precomputed Features Format

Each `{video_id}.pt` file should be a dict saved with `torch.save()`:
```python
{
    "entity_labels": torch.Tensor,    # [N] int — entity class indices
    "tube_features": torch.Tensor,    # [N, T, D] float — tube representations
    "relations": [                     # list of annotation dicts
        {"subject_idx": int, "object_idx": int, "relation_cat": int,
         "start_frame": int, "end_frame": int},
        ...
    ],
    "num_frames": int,
}
```

## Training

### Synthetic debug

```bash
python scripts/run_synthetic_debug.py
# or
python scripts/train.py --config configs/synthetic.yaml
```

### OpenPVSG

```bash
# With contrastive learning (paper method)
python scripts/train.py --config configs/openpvsg.yaml

# Edit configs/openpvsg.yaml to change:
#   tube_encoder_type: transformer  or  convolution
#   use_contrastive: true/false
#   similarity_method: optimal_transport / pooling_cosine / pooling_l2
```

### PSG4D

```bash
# RGB-D
python scripts/train.py --config configs/psg4d_rgbd.yaml

# Point cloud
python scripts/train.py --config configs/psg4d_pointcloud.yaml
```

### Key hyperparameters (from paper)

| Parameter | Value | Config key |
|-----------|-------|------------|
| Relation classifier LR | 1e-3 | `learning_rate` |
| Batch size | 32 | `batch_size` |
| Segmentation LR (IPS+T/VPS) | 1e-4 | seg config |
| Segmentation weight decay | 0.05 | seg config |
| Gradient clip (seg) | 0.01 | `gradient_clip_norm` |
| Motion threshold γ | 9.0 | `gamma` |
| OT margin α | 10.0 | `alpha` |
| Sinkhorn iterations | 1000 | `sinkhorn_iterations` |
| PSG4D relation epochs | 100 | `num_epochs` |
| PSG4D RGB-D seg epochs | 12 | seg config |
| PSG4D pointcloud seg epochs | 200 | seg config |

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/synthetic.yaml \
    --checkpoint checkpoints/synthetic/checkpoint_final.pt

# Produces table with R@20, R@50, R@100, mR@20, mR@50, mR@100
# at vIoU thresholds 0.5 and 0.1
```

## Plugging in Precomputed Segmentation

The relation model is decoupled from segmentation. To use your own segmentation outputs:

1. Run your segmentation model (Mask2Former, Video K-Net, etc.) on the target dataset.
2. Save tube features per video as `.pt` files (see format above).
3. Place them in `{data_root}/precomputed_features/`.
4. Set `use_precomputed: true` (default) in the dataset loader.

For direct integration with segmentation models, see wrapper classes in `src/motion_aware_tpsgg/models/segmentation.py`.

## Implementation Fidelity

### Faithful to the paper

- **Contrastive framework** — InfoNCE loss with positive sampling (same category, different video), shuffle-based negatives (Eq. 6), triplet-based hard negatives with multinomial sampling
- **Optimal transport** — Sinkhorn algorithm for mask tube distance (Algorithm 1), cosine distance cost, distance-to-similarity conversion (Eq. 9)
- **Motion strength** — Sobel filter on optical flow magnitude, median over entity masks, threshold γ selection
- **Relation classifier** — Tube encoder → global pooling → MLP (Eq. 2, 3)
- **Evaluation metrics** — R@K and mR@K with vIoU-based tube matching at thresholds 0.5 and 0.1
- **All hyperparameters** — γ=9.0, α=10.0, N_iter=1000, optimizer choices, learning rates, epochs as reported

### Assumptions and placeholders

- **Segmentation models** — Wrapper interfaces provided for Mask2Former+UniTrack, Video K-Net, PSG4DFormer. Actual model code requires separate installation of detectron2/mmdet. The relation pipeline runs independently using precomputed features.
- **Optical flow** — `estimate_motion_strength()` accepts precomputed flow magnitude. Computing flow from raw video requires an external optical flow model (e.g., RAFT).
- **Dataset loaders** — OpenPVSG and PSG4D loaders parse the expected annotation format. Minor differences from the actual annotation schema may need adaptation (the exact JSON keys vary by dataset version).
- **Sinkhorn loop over s** — Algorithm 1 iterates over partial transport masses s=1..min(T_i,T_j). For long tubes this can be slow; the full-mass case (s=min) dominates in practice.
- **Contrastive index rebuild** — The triplet index for contrastive sampling is built once before training. For large datasets, this could be made lazy or epoch-wise.

## Tests

```bash
# Run all 35 tests
pytest tests/test_all.py -v
```

Tests cover: vIoU computation, R@K/mR@K metrics, Sinkhorn OT distance, positive/negative sampling, synthetic dataloader, model forward passes, loss computation, and one full training step.

## Citation

```bibtex
@inproceedings{nguyen2025motion,
  title={Motion-aware Contrastive Learning for Temporal Panoptic Scene Graph Generation},
  author={Nguyen, Thong Thanh and Wu, Xiaobao and Bin, Yi and Nguyen, Cong-Duy and Ng, See-Kiong and Luu, Anh Tuan},
  booktitle={AAAI},
  year={2025}
}
```
