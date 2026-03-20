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
    parser.add_argument("--index", type=int, required=True,
                        help="Index into NaturalBench dataset to segment")
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
    print(f"  overlay -> {out_path}")


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
    print(f"  segmap  -> {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    device = args.device

    # 1. Load dataset
    print("Loading NaturalBench dataset...")
    ds = load_naturalbench_dataset("test")

    if args.index < 0 or args.index >= len(ds):
        raise ValueError(f"Index {args.index} out of range (dataset size = {len(ds)})")

    row = ds[args.index]
    question = row["Question"]
    answer = row["Answer"]
    print(f"Dataset row {args.index}:")
    print(f"  Q: {question}")
    print(f"  A: {answer}")

    # 2. Load the image
    image = row["Image"]
    # NaturalBench usually stores PIL.Image in the 'Image' column
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    width, height = image.size
    print(f"Image size: {width}x{height}")

    # 3. Load segmentation model
    model_id = "facebook/mask2former-swin-base-coco-panoptic"
    print(f"Loading model: {model_id}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device)
    model.eval()

    # 4. Run segmentation
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Panoptic segmentation post-processing
    panoptic = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(height, width)]
    )[0]

    seg_map = panoptic["segmentation"].cpu().numpy()  # H x W
    segments_info = panoptic["segments_info"]

    # 5. Extract masks + metadata
    id2label = getattr(processor, "id2label", None)

    masks = []
    meta = []

    MIN_AREA = 50  # remove tiny segments

    for seg in segments_info:
        seg_id = seg["id"]
        mask = seg_map == seg_id
        area = int(mask.sum())
        if area < MIN_AREA:
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

    # 6. Save results (masks, meta, visualizations)
    out_dir = "m2f_masks"
    os.makedirs(out_dir, exist_ok=True)

    base = f"nb_{args.index}"

    masks_path = os.path.join(out_dir, f"{base}_masks.npz")
    meta_path = os.path.join(out_dir, f"{base}_meta.json")
    overlay_path = os.path.join(out_dir, f"{base}_overlay.png")
    segmap_path = os.path.join(out_dir, f"{base}_segmap.png")

    # Save masks + metadata
    np.savez_compressed(masks_path, masks=masks_np)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {masks_np.shape[0]} masks for dataset index {args.index}")
    print(f"  masks   -> {masks_path}")
    print(f"  meta    -> {meta_path}")

    # Save segmentation map visualization
    save_segmentation_map(seg_map, segmap_path)

    # Save red overlay visualization (union of all kept masks)
    if masks_np.shape[0] > 0:
        union_mask = masks_np.any(axis=0)
    else:
        # If no masks passed MIN_AREA, just create an all-false mask (no overlay)
        union_mask = np.zeros((height, width), dtype=bool)

    save_red_overlay(image, union_mask, overlay_path, alpha=0.5)


if __name__ == "__main__":
    main()
