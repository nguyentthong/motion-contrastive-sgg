#!/usr/bin/env python3
"""Prepare PSG4D annotations into train/val splits.

Usage:
    uv run python scripts/prepare_psg4d.py --root data/psg4d/psg4d_gta
    uv run python scripts/prepare_psg4d.py --root data/psg4d/psg4d_hoi
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    ann_dir = root / "annotations"

    # Check for existing annotations
    if (ann_dir / "train.json").exists() and (ann_dir / "val.json").exists():
        print("Train/val splits already exist.")
        return

    # Look for a main annotation file
    main_ann = None
    for name in ["psg4d.json", "annotations.json", "data.json"]:
        if (ann_dir / name).exists():
            main_ann = ann_dir / name
            break

    if main_ann is None:
        print(f"No annotation file found in {ann_dir}")
        print("Expected one of: psg4d.json, annotations.json, data.json")
        print("Please download from: https://github.com/Jingkang50/4D-PSG")
        return

    with open(main_ann) as f:
        data = json.load(f)

    all_data = data.get("data", [])
    n = len(all_data)
    split_idx = int(0.8 * n)

    with open(ann_dir / "train.json", "w") as f:
        json.dump({"data": all_data[:split_idx]}, f)
    with open(ann_dir / "val.json", "w") as f:
        json.dump({"data": all_data[split_idx:]}, f)

    print(f"Train: {split_idx}, Val: {n - split_idx}")


if __name__ == "__main__":
    main()
