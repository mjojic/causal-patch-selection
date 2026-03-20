#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# ---------------------------------------------------------------------
# HF snapshot helpers
# ---------------------------------------------------------------------
def resolve_hf_snapshot_dir(model_dir: str) -> str:
    """
    Resolve HuggingFace cache directory to actual snapshot directory.
    Handles both direct snapshot paths and hub cache layouts.
    """
    p = Path(model_dir)
    if (p / "config.json").exists():
        return str(p)
    snaps = sorted(glob.glob(str(p / "snapshots" / "*")), key=lambda x: os.path.getmtime(x))
    if not snaps:
        raise FileNotFoundError(f"Could not find snapshots under: {model_dir}")
    snap = Path(snaps[-1])
    if not (snap / "config.json").exists():
        raise FileNotFoundError(f"Snapshot missing config.json: {snap}")
    return str(snap)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_dir",
        type=str,
        default="/mnt/arc/zhaonan2/blind_project/datasets/seed_bench/seed_images",
        help="Directory containing seed_bench .jpg images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/arc/mjojic/causal-patch-selection/segment_patches/seed_bench",
        help="Directory to save masks and metadata",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Inclusive start index of image list to process",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Exclusive end index of image list to process (-1 = until end)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/shared/shared_hf_home/hub/models--facebook--mask2former-swin-base-coco-panoptic",
        help="Path to Mask2Former model (local path or HuggingFace repo_id)",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=50,
        help="Minimum area for a segment to be kept",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device string, e.g. 'cuda:0'",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def save_segmentation_map(seg_map: np.ndarray, out_path: str):
    """
    Save a pseudo-colored segmentation map for visualization.
    seg_map: H x W, integer segment id per pixel.
    """
    h, w = seg_map.shape
    ids = np.unique(seg_map)
    rng = np.random.default_rng(1234)

    id2color = {}
    for seg_id in ids:
        color = rng.integers(low=64, high=255, size=3, dtype=np.uint8)
        id2color[int(seg_id)] = color

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for seg_id, color in id2color.items():
        rgb[seg_map == seg_id] = color

    out_img = Image.fromarray(rgb, mode="RGB")
    out_img.save(out_path)


# ---------------------------------------------------------------------
# Process single image (path + stem)
# ---------------------------------------------------------------------
def process_image_path(
    image_path: str,
    base_name: str,
    model,
    processor,
    device: str,
    out_dir: str,
    min_area: int,
) -> bool:
    """
    Process a single image from path:
    - Load image
    - Run segmentation
    - Save masks, metadata, and visualizations using base_name (e.g. stem).

    Returns True if successful, False otherwise.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Run segmentation
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Panoptic segmentation post-processing
        panoptic = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]

        seg_map = panoptic["segmentation"].cpu().numpy()  # H x W
        segments_info = panoptic["segments_info"]

        # Extract masks + metadata
        id2label = getattr(processor, "id2label", None)

        masks = []
        meta = []

        for seg in segments_info:
            seg_id = seg["id"]
            mask = seg_map == seg_id
            area = int(mask.sum())
            if area < min_area:
                continue

            label_id = seg["label_id"]
            label_name = id2label.get(label_id, str(label_id)) if id2label else str(label_id)

            masks.append(mask)
            meta.append({
                "id": int(seg_id),
                "label_id": int(label_id),
                "label": label_name,
                "score": float(seg.get("score", 0.0)),
                "isthing": bool(seg.get("isthing", False)),
                "area": area,
            })

        if masks:
            masks_np = np.stack(masks, axis=0).astype(bool)
        else:
            masks_np = np.zeros((0, height, width), dtype=bool)

        # Save results using base_name (filename stem)
        masks_path = os.path.join(out_dir, f"{base_name}_masks.npz")
        meta_path = os.path.join(out_dir, f"{base_name}_meta.json")
        segmap_path = os.path.join(out_dir, f"{base_name}_segmap.png")

        np.savez_compressed(masks_path, masks=masks_np)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        save_segmentation_map(seg_map, segmap_path)

        print(f"[{base_name}] Saved {masks_np.shape[0]} masks ({width}x{height})")
        return True

    except Exception as e:
        print(f"[{base_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover .jpg images and sort for deterministic order
    image_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))
    n_total = len(image_paths)
    print(f"Found {n_total} images in {args.images_dir}")

    start_idx = max(0, args.start_idx)
    end_idx = n_total if args.end_idx < 0 else min(args.end_idx, n_total)
    if start_idx >= end_idx:
        print(f"Empty index range: start_idx={start_idx}, end_idx={end_idx}")
        return
    image_paths = image_paths[start_idx:end_idx]
    print(f"Processing image indices [{start_idx}, {end_idx}) ({len(image_paths)} images)")

    # Resolve model path and load segmentation model
    print(f"Resolving model path: {args.model_path}")
    try:
        snapshot_path = resolve_hf_snapshot_dir(args.model_path)
        print(f"Resolved snapshot: {snapshot_path}")
    except (FileNotFoundError, ValueError):
        print(f"Could not resolve snapshot, using path directly: {args.model_path}")
        snapshot_path = args.model_path

    print(f"Loading model from: {snapshot_path}")
    try:
        processor = AutoImageProcessor.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            snapshot_path,
            local_files_only=True,
        ).to(device)
    except (OSError, ValueError) as e:
        print(f"Failed with local_files_only=True, trying without: {e}")
        processor = AutoImageProcessor.from_pretrained(snapshot_path)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(snapshot_path).to(device)

    model.eval()
    print("Model loaded successfully")

    successful = 0
    failed = 0

    for image_path in image_paths:
        base_name = Path(image_path).stem

        print("\n" + "=" * 80)
        print(f"Processing {base_name}")
        print("=" * 80)

        if process_image_path(
            image_path,
            base_name,
            model,
            processor,
            device,
            args.output_dir,
            args.min_area,
        ):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
