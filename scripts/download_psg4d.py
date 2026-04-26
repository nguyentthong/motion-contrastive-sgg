#!/usr/bin/env python3
"""Download and prepare the PSG4D dataset.

PSG4D project: https://github.com/Jingkang50/4D-PSG
Paper: "4D Panoptic Scene Graph Generation" (NeurIPS 2024)

Usage:
    uv run python scripts/download_psg4d.py --output data/psg4d
"""

import argparse
from pathlib import Path


EXPECTED_STRUCTURE = """
Expected directory structure:

data/psg4d/
├── psg4d_gta/
│   ├── annotations/
│   │   ├── train.json
│   │   └── val.json
│   ├── rgb/
│   │   └── {video_id}/
│   │       └── *.jpg
│   ├── depth/
│   │   └── {video_id}/
│   │       └── *.png
│   ├── pointclouds/
│   │   └── {video_id}/
│   │       └── *.ply
│   └── precomputed_features/
│       ├── rgbd/
│       │   └── {video_id}.pt
│       └── pointcloud/
│           └── {video_id}.pt
└── psg4d_hoi/
    └── (same structure as psg4d_gta)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/psg4d")
    args = parser.parse_args()

    root = Path(args.output)

    for split in ["psg4d_gta", "psg4d_hoi"]:
        for subdir in ["annotations", "rgb", "depth", "pointclouds",
                        "precomputed_features/rgbd", "precomputed_features/pointcloud"]:
            (root / split / subdir).mkdir(parents=True, exist_ok=True)

    print("PSG4D Dataset Preparation")
    print("=" * 50)
    print(EXPECTED_STRUCTURE)
    print("Steps:")
    print("1. Clone: git clone https://github.com/Jingkang50/4D-PSG")
    print("2. Follow their instructions for downloading PSG4D-GTA and PSG4D-HOI")
    print("3. Place annotations in the appropriate directories")
    print("4. Run segmentation and save precomputed features")
    print()
    print(f"Directory structure created at: {root}")


if __name__ == "__main__":
    main()
