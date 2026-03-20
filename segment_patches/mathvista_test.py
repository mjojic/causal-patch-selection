#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

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


def extract_repo_id_from_cache_path(cache_path: str) -> str:
    """
    Extract HuggingFace repo_id from cache path format.
    
    Cache format: models--namespace--repo-name
    Repo format: namespace/repo-name
    
    Example:
        models--facebook--mask2former-swin-base-coco-panoptic
        -> facebook/mask2former-swin-base-coco-panoptic
    """
    # Get the directory name (last component of path)
    dir_name = os.path.basename(cache_path.rstrip('/'))
    
    # Check if it's in cache format (starts with models--)
    if dir_name.startswith("models--"):
        # Remove "models--" prefix
        repo_part = dir_name[8:]  # len("models--") = 8
        # Replace "--" with "/"
        repo_id = repo_part.replace("--", "/")
        return repo_id
    
    # If it's already a repo_id format or direct path, return as-is
    # (will be validated by HuggingFace)
    return dir_name


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Core parameters
    parser.add_argument(
        "--image_path",
        type=str,
        default="/mnt/arc/mjojic/mathvista_1.png",
        help="Path to input image to segment",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/arc/mjojic/segmap_mathvista_1.png",
        help="Path to save segmentation map",
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
        default="cuda:2",
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
    # Simple pseudo-color: hash segment id to RGB
    ids = np.unique(seg_map)
    # avoid id == 0 always black; we still color it
    rng = np.random.default_rng(1234)

    id2color = {}
    for seg_id in ids:
        # random bright-ish color
        color = rng.integers(low=64, high=255, size=3, dtype=np.uint8)
        id2color[int(seg_id)] = color

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for seg_id, color in id2color.items():
        rgb[seg_map == seg_id] = color

    out_img = Image.fromarray(rgb, mode="RGB")
    out_img.save(out_path)


# ---------------------------------------------------------------------
# Process single image
# ---------------------------------------------------------------------
def process_image(image_path: str, output_path: str, model, processor, device: str, min_area: int) -> bool:
    """
    Process a single image:
    - Load image from file path
    - Run segmentation
    - Save segmentation map visualization
    
    Returns True if successful, False otherwise
    """
    try:
        # Load the image
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

        # Save segmentation map visualization
        save_segmentation_map(seg_map, output_path)

        print(f"Saved {masks_np.shape[0]} masks ({width}x{height}) to {output_path}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    device = args.device

    # Check input image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Input image not found: {args.image_path}")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Resolve model path and load segmentation model
    print(f"Checking model path: {args.model_path}")
    
    # Check if model exists locally
    model_exists = False
    snapshot_path = None
    repo_id = None
    
    # First, check if it's a local directory path
    if os.path.exists(args.model_path):
        try:
            snapshot_path = resolve_hf_snapshot_dir(args.model_path)
            # Verify it's actually a valid model directory
            if os.path.exists(os.path.join(snapshot_path, "config.json")):
                model_exists = True
                print(f"Found model in cache: {snapshot_path}")
        except (FileNotFoundError, ValueError):
            pass
    
    # If model doesn't exist locally, extract repo_id and download it
    if not model_exists:
        # Extract repo_id from cache path format
        repo_id = extract_repo_id_from_cache_path(args.model_path)
        print(f"Model not found in cache. Downloading from HuggingFace Hub: {repo_id}")
        snapshot_path = repo_id  # Use extracted repo_id for download
    else:
        snapshot_path = resolve_hf_snapshot_dir(args.model_path)
    
    print(f"Loading model from: {snapshot_path}")
    
    # Load with local_files_only only if we confirmed it exists locally
    if model_exists:
        print("Loading from local cache...")
        processor = AutoImageProcessor.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            snapshot_path,
            local_files_only=True,
        ).to(device)
    else:
        print(f"Downloading model from HuggingFace Hub (repo_id: {repo_id})...")
        processor = AutoImageProcessor.from_pretrained(snapshot_path)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(snapshot_path).to(device)
    
    model.eval()
    print("Model loaded successfully")

    # Process the image
    print("\n" + "=" * 80)
    print(f"Processing image: {args.image_path}")
    print("=" * 80)

    if process_image(args.image_path, args.output_path, model, processor, device, args.min_area):
        print("Segmentation completed successfully!")
    else:
        print("Segmentation failed!")


if __name__ == "__main__":
    main()
