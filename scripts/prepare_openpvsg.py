#!/usr/bin/env python3
"""Prepare OpenPVSG annotations: split into train/val JSON files.

Reads pvsg.json and creates train.json / val.json with the proper format
for the dataset loader.

Usage:
    uv run python scripts/prepare_openpvsg.py --root data/openpvsg
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/openpvsg")
    args = parser.parse_args()

    root = Path(args.root)
    ann_path = root / "annotations" / "pvsg.json"

    if not ann_path.exists():
        print(f"Annotation file not found: {ann_path}")
        print("Please download pvsg.json from the OpenPVSG project first.")
        print("See: https://github.com/Jingkang50/OpenPSG")
        return

    with open(ann_path) as f:
        pvsg = json.load(f)

    # The OpenPVSG format has a 'data' key with video annotations
    # and 'split' information. Adjust based on actual format.
    all_data = pvsg.get("data", [])
    train_split = pvsg.get("train_ids", [])
    val_split = pvsg.get("val_ids", [])

    # If splits aren't defined, do 80/20
    if not train_split and not val_split:
        n = len(all_data)
        split_idx = int(0.8 * n)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    else:
        train_ids_set = set(train_split)
        val_ids_set = set(val_split)
        train_data = [d for d in all_data if d.get("video_id") in train_ids_set]
        val_data = [d for d in all_data if d.get("video_id") in val_ids_set]

    train_out = root / "annotations" / "train.json"
    val_out = root / "annotations" / "val.json"

    with open(train_out, "w") as f:
        json.dump({"data": train_data}, f)
    with open(val_out, "w") as f:
        json.dump({"data": val_data}, f)

    print(f"Train: {len(train_data)} videos -> {train_out}")
    print(f"Val: {len(val_data)} videos -> {val_out}")


if __name__ == "__main__":
    main()
