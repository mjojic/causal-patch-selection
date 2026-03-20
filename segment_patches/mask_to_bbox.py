#!/usr/bin/env python3
"""
Convert fine-grained segmentation masks + selected object indices into a single
JSON file in the format expected by LVR training (train_lvr.py / lvr_sft_dataset.py).

Inputs:
  - NPZ mask files: nb_{image_idx}_masks.npz (masks array shape (num_masks, H, W))
  - Consensus JSONs: contain image_idx, question, gold_answer, and a list of
    selected mask indices (key configurable via --mask_indices_key)

Output:
  - One JSON file: list of LVR samples with "image", "bboxes", "conversations".
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert masks + consensus JSONs to LVR-format bbox JSON."
    )
    parser.add_argument(
        "--masks_directory",
        type=str,
        required=True,
        help="Directory containing nb_{image_idx}_masks.npz files.",
    )
    parser.add_argument(
        "--consensus_json_directory",
        type=str,
        required=True,
        help="Directory containing consensus JSON files (e.g. consensus_1.json).",
    )
    parser.add_argument(
        "--bbox_json_directory",
        type=str,
        required=True,
        help="Directory where the output JSON file will be written.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="lvr_data.json",
        help="Output JSON filename (default: lvr_data.json).",
    )
    parser.add_argument(
        "--mask_indices_key",
        type=str,
        default="consensus_mask_indices",
        help="JSON key for the list of selected mask indices (default: consensus_mask_indices).",
    )
    parser.add_argument(
        "--image_filename_pattern",
        type=str,
        default="nb_{image_idx}.jpg",
        help="Pattern for image path in output; must contain {image_idx} (default: nb_{image_idx}.jpg).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing NPZ or invalid mask index; otherwise skip with warning.",
    )
    parser.add_argument(
        "--no_indent",
        action="store_true",
        help="Write JSON without indentation (smaller file).",
    )
    return parser.parse_args()


def discover_consensus_files(consensus_dir: str, mask_indices_key: str):
    """Discover consensus JSON files that contain image_idx and mask_indices_key."""
    consensus_path = Path(consensus_dir)
    if not consensus_path.is_dir():
        raise FileNotFoundError(f"Consensus directory not found: {consensus_dir}")

    found = []
    for p in consensus_path.iterdir():
        if p.suffix.lower() != ".json":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("Skip %s: %s", p.name, e)
            continue
        if "image_idx" not in data:
            logging.debug("Skip %s: missing image_idx", p.name)
            continue
        if mask_indices_key not in data:
            logging.warning("Skip %s: missing key %s", p.name, mask_indices_key)
            continue
        found.append((p, data))
    return found


def masks_to_bboxes(masks: np.ndarray, indices: list, strict: bool) -> list:
    """
    For each mask index, compute normalized [x_min, y_min, x_max, y_max] in [0, 1].
    Skips empty or out-of-range masks (or raises if strict).
    """
    n_masks, height, width = masks.shape
    bboxes = []
    for idx in indices:
        if idx < 0 or idx >= n_masks:
            msg = f"Mask index {idx} out of range [0, {n_masks})"
            if strict:
                raise ValueError(msg)
            logging.warning(msg)
            continue
        mask = masks[idx]
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            logging.warning("Empty mask at index %s; skipping", idx)
            continue
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        # Normalize by image dimensions and clamp to [0, 1]
        x_min_norm = max(0.0, min(1.0, x_min / width))
        x_max_norm = max(0.0, min(1.0, x_max / width))
        y_min_norm = max(0.0, min(1.0, y_min / height))
        y_max_norm = max(0.0, min(1.0, y_max / height))
        bboxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
    return bboxes


def build_lvr_sample(
    image_idx: int,
    question: str,
    gold_answer: str,
    bboxes: list,
    image_filename_pattern: str,
) -> dict:
    """Build one LVR-format sample: image path, bboxes, conversations with <image> and <lvr>."""
    image_path = image_filename_pattern.format(image_idx=image_idx)
    # One <lvr> per bbox; order must match bboxes (replace_lvr_tokens zips by position).
    lvr_placeholders = " ".join(["<lvr>"] * len(bboxes))
    user_value = f"<image>\n{question} {lvr_placeholders}"
    conversations = [
        {"from": "human", "value": user_value},
        {"from": "gpt", "value": gold_answer},
    ]
    return {
        "image": image_path,
        "bboxes": bboxes,
        "conversations": conversations,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    masks_dir = Path(args.masks_directory)
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {args.masks_directory}")

    # Discover consensus files
    consensus_list = discover_consensus_files(
        args.consensus_json_directory, args.mask_indices_key
    )
    logging.info("Found %d consensus JSON(s) with %s", len(consensus_list), args.mask_indices_key)

    lvr_samples = []
    skipped_no_npz = 0
    skipped_no_bboxes = 0

    for _path, data in consensus_list:
        image_idx = data["image_idx"]
        question = data.get("question", "")
        gold_answer = data.get("gold_answer", "")
        mask_indices = data[args.mask_indices_key]
        if not isinstance(mask_indices, list):
            logging.warning("%s: %s is not a list; skipping", _path.name, args.mask_indices_key)
            continue

        npz_path = masks_dir / f"nb_{image_idx}_masks.npz"
        if not npz_path.exists():
            if args.strict:
                raise FileNotFoundError(f"Missing NPZ: {npz_path}")
            logging.warning("Missing NPZ for image_idx %s: %s", image_idx, npz_path)
            skipped_no_npz += 1
            continue

        with np.load(npz_path, allow_pickle=False) as npz:
            masks = npz["masks"]
        if masks.size == 0:
            logging.warning("Empty masks array for image_idx %s", image_idx)
            skipped_no_bboxes += 1
            continue

        bboxes = masks_to_bboxes(masks, mask_indices, args.strict)
        if not bboxes:
            logging.warning("No valid bboxes for image_idx %s (file %s)", image_idx, _path.name)
            skipped_no_bboxes += 1
            continue

        sample = build_lvr_sample(
            image_idx=image_idx,
            question=question,
            gold_answer=gold_answer,
            bboxes=bboxes,
            image_filename_pattern=args.image_filename_pattern,
        )
        lvr_samples.append(sample)

    out_dir = Path(args.bbox_json_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    with open(out_path, "w", encoding="utf-8") as f:
        if args.no_indent:
            json.dump(lvr_samples, f, ensure_ascii=False)
        else:
            json.dump(lvr_samples, f, indent=2, ensure_ascii=False)

    logging.info(
        "Wrote %d LVR samples to %s (skipped: %d no NPZ, %d no bboxes).",
        len(lvr_samples),
        out_path,
        skipped_no_npz,
        skipped_no_bboxes,
    )


if __name__ == "__main__":
    main()
