#!/usr/bin/env python3

import argparse
import os
import json

import numpy as np
import torch
from PIL import Image
import datasets
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# ---------------------------------------------------------
# NaturalBench loader
# ---------------------------------------------------------
def load_naturalbench_dataset(split: str = "test"):
    ds = datasets.load_dataset("BaiqiL/NaturalBench-lmms-eval", split=split)
    return ds


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True,
                        help="Cuda device string, e.g. 'cuda:0'")
    parser.add_argument("--start_idx", type=int, required=True,
                        help="Start index (inclusive) into NaturalBench dataset")
    parser.add_argument("--end_idx", type=int, required=True,
                        help="End index (exclusive) into NaturalBench dataset")
    parser.add_argument("--output_dir", type=str, default="naturalbench_objects",
                        help="Directory to save masks and metadata")
    parser.add_argument("--min_area", type=int, default=50,
                        help="Minimum area for a segment to be kept")
    parser.add_argument("--model_id", type=str, 
                        default="facebook/mask2former-swin-base-coco-panoptic",
                        help="Hugging Face model ID for segmentation")
    return parser.parse_args()


# ---------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------
def save_red_overlay(image: Image.Image, union_mask: np.ndarray, out_path: str,
                     alpha: float = 0.5):
    """
    Save an image where all True pixels in union_mask are shaded red
    on top of the original image.
    alpha: how strong the red overlay is (0..1)
    """
    # image: PIL RGB, union_mask: (H, W) bool
    img_np = np.array(image).astype(np.float32)  # H,W,3

    # Create red image
    red = np.zeros_like(img_np)
    red[..., 0] = 255.0  # R channel

    mask = union_mask.astype(bool)
    mask_3 = np.stack([mask] * 3, axis=-1)  # H,W,3

    # alpha blend: (1 - alpha) * orig + alpha * red for mask pixels
    blended = img_np.copy()
    blended[mask_3] = (1.0 - alpha) * img_np[mask_3] + alpha * red[mask_3]

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(blended)
    out_img.save(out_path)


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


# ---------------------------------------------------------
# Process single image
# ---------------------------------------------------------
def process_image(idx, row, model, processor, device, out_dir, min_area):
    """
    Process a single image from the dataset:
    - Extract image
    - Run segmentation
    - Save masks, metadata, and visualizations
    
    Returns True if successful, False otherwise
    """
    try:
        question = row["Question"]
        answer = row["Answer"]
        
        # Load the image
        image = row["Image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

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

        # Save results
        base = f"nb_{idx}"

        masks_path = os.path.join(out_dir, f"{base}_masks.npz")
        meta_path = os.path.join(out_dir, f"{base}_meta.json")
        overlay_path = os.path.join(out_dir, f"{base}_overlay.png")
        segmap_path = os.path.join(out_dir, f"{base}_segmap.png")

        # Save masks + metadata
        np.savez_compressed(masks_path, masks=masks_np)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Save segmentation map visualization
        save_segmentation_map(seg_map, segmap_path)

        # Save red overlay visualization (union of all kept masks)
        if masks_np.shape[0] > 0:
            union_mask = masks_np.any(axis=0)
        else:
            union_mask = np.zeros((height, width), dtype=bool)

        save_red_overlay(image, union_mask, overlay_path, alpha=0.5)

        print(f"[{idx}] Saved {masks_np.shape[0]} masks ({width}x{height})")
        return True

    except Exception as e:
        print(f"[{idx}] ERROR: {e}")
        return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    device = args.device

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading NaturalBench dataset...")
    ds = load_naturalbench_dataset("test")
    dataset_size = len(ds)
    print(f"Dataset size: {dataset_size}")

    # Validate indices
    if args.start_idx < 0 or args.start_idx >= dataset_size:
        raise ValueError(f"start_idx {args.start_idx} out of range [0, {dataset_size})")
    if args.end_idx <= args.start_idx or args.end_idx > dataset_size:
        raise ValueError(f"end_idx {args.end_idx} must be > start_idx and <= {dataset_size}")

    print(f"Processing indices [{args.start_idx}, {args.end_idx})")

    # Load segmentation model
    print(f"Loading model: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()

    # Process each image in range
    successful = 0
    failed = 0

    for idx in range(args.start_idx, args.end_idx):
        row = ds[idx]
        if process_image(idx, row, model, processor, device, args.output_dir, args.min_area):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
