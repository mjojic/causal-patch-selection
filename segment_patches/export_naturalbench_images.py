#!/usr/bin/env python3
"""
Export NaturalBench images from the HuggingFace dataset to a directory as nb_0.jpg,
nb_1.jpg, ... so that LVR training can resolve paths like "nb_1195.jpg" via
--image_folder.

The dataset is stored under the HF cache (e.g. BaiqiL___natural_bench-lmms-eval);
this script loads it via datasets.load_dataset with cache_dir and saves each
row's Image as nb_{Index}.jpg (row index = Index column).
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export NaturalBench images to nb_{index}.jpg for LVR image_folder."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write nb_0.jpg, nb_1.jpg, ...",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/shared/shared_hf_home",
        help="HuggingFace cache root (default: /mnt/shared/shared_hf_home).",
    )
    parser.add_argument(
        "--lvr_json",
        type=str,
        default=None,
        help="Optional path to LVR bbox JSON; if set, only export image indices that appear in this file.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=("jpg", "jpeg", "png"),
        help="Image format (default: jpg).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test).",
    )
    return parser.parse_args()


def image_indices_from_lvr_json(path: str) -> set:
    """Return set of image indices (e.g. 1195) extracted from 'image' keys like 'nb_1195.jpg'."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    indices = set()
    for item in data:
        img = item.get("image", "")
        if img.startswith("nb_") and "." in img:
            try:
                idx = int(img.split("nb_")[1].split(".")[0])
                indices.add(idx)
            except ValueError:
                continue
    return indices


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only_indices = None
    if args.lvr_json:
        only_indices = image_indices_from_lvr_json(args.lvr_json)
        logging.info("Exporting only %d indices present in %s", len(only_indices), args.lvr_json)

    logging.info("Loading NaturalBench (split=%s, cache_dir=%s)...", args.split, args.cache_dir)
    ds = load_dataset(
        "BaiqiL/NaturalBench-lmms-eval",
        split=args.split,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    n = len(ds)
    ext = ".jpg" if args.format in ("jpg", "jpeg") else f".{args.format}"

    exported = 0
    for i in range(n):
        if only_indices is not None and i not in only_indices:
            continue
        row = ds[i]
        img = row["Image"]
        if img is None:
            logging.warning("Row %d: no image", i)
            continue
        path = out_dir / f"nb_{i}{ext}"
        if args.format in ("jpg", "jpeg"):
            img.save(path, format="JPEG", quality=95)
        else:
            img.save(path, format="PNG")
        exported += 1
        if exported <= 3 or exported % 500 == 0:
            logging.info("Exported %d -> %s", i, path.name)

    logging.info("Done. Exported %d images to %s", exported, out_dir)
    logging.info("Use --image_folder %s when running train_lvr.py", out_dir.resolve())


if __name__ == "__main__":
    main()
