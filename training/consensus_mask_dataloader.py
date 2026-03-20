#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import datasets

# Import pixel-to-token conversion utilities
try:
    from pixel_to_token_utils import (
        llm_image_token_grid_from_inputs,
        pixel_mask_to_llm_token_mask_from_inputs,
        verify_token_mask_matches_vision_span,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pixel_to_token_utils import (
        llm_image_token_grid_from_inputs,
        pixel_mask_to_llm_token_mask_from_inputs,
        verify_token_mask_matches_vision_span,
    )

from qwen_vl_utils import process_vision_info


class ConsensusDatasetIndex:
    """Fast indexing structure for consensus results."""

    def __init__(self, consensus_dir: str):
        self.consensus_dir = Path(consensus_dir)
        self.index: Dict[int, Path] = {}
        self._build_index()

    def _build_index(self):
        json_files = sorted(self.consensus_dir.glob("consensus_*.json"))
        for json_path in json_files:
            match = re.search(r"consensus_(\d+)\.json$", json_path.name)
            if match:
                idx = int(match.group(1))
                self.index[idx] = json_path
        print(f"[ConsensusDatasetIndex] Indexed {len(self.index)} consensus results")

    def get_path(self, idx: int) -> Optional[Path]:
        return self.index.get(idx)

    def get_all_indices(self) -> List[int]:
        return sorted(self.index.keys())

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, idx: int) -> bool:
        return idx in self.index


def load_masks_for_index(masks_dir: str, idx: int) -> Optional[np.ndarray]:
    """
    Load original segmentation masks for a given image index.
    Returns: masks array of shape (N, H, W) bool, or None if not found.
    """
    mask_path = os.path.join(masks_dir, f"nb_{idx}_masks.npz")
    if not os.path.exists(mask_path):
        return None
    data = np.load(mask_path)
    masks = data["masks"]  # (N, H, W)
    if masks.dtype != bool:
        masks = masks.astype(bool)
    return masks


class ConsensusMaskDataset(Dataset):
    """
    Returns per item:
      - image: PIL.Image (possibly downscaled if max_side>0)
      - token_mask: torch.BoolTensor (h_llm, w_llm)   [if precompute_token_masks=True]
      - token_grid_hw: (h_llm, w_llm)                 [per-sample]
      - question / answer choices / gold letter / etc.
    """

    def __init__(
        self,
        consensus_dir: str,
        model=None,
        processor=None,
        naturalbench_split: str = "test",
        masks_dir: Optional[str] = None,
        image_transform: Optional[Callable] = None,
        cache_images: bool = False,
        cache_masks: bool = True,
        use_precomputed_masks: bool = True,
        precompute_token_masks: bool = True,
        grid_prompt_text: str = "dummy",
        verbose_every: int = 25,
        max_side: int = 0,  # if >0, downscale so max(W,H)=max_side
    ):
        self.consensus_dir = Path(consensus_dir)
        self.model = model
        self.processor = processor
        self.naturalbench_split = naturalbench_split
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.image_transform = image_transform
        self.cache_images = bool(cache_images)
        self.cache_masks = bool(cache_masks)
        self.use_precomputed_masks = bool(use_precomputed_masks)
        self.precompute_token_masks = bool(precompute_token_masks)
        self.grid_prompt_text = grid_prompt_text
        self.verbose_every = int(verbose_every)
        self.max_side = int(max_side) if max_side else 0

        # Build index + list of available NB indices
        self.index = ConsensusDatasetIndex(consensus_dir)
        self.indices = self.index.get_all_indices()

        # Load NaturalBench
        print(f"[ConsensusMaskDataset] Loading NaturalBench {naturalbench_split} split...")
        self.naturalbench = datasets.load_dataset(
            "BaiqiL/NaturalBench-lmms-eval",
            split=naturalbench_split,
        )

        # Caches
        self._image_cache: Dict[int, Image.Image] = {}
        self._mask_cache: Dict[int, np.ndarray] = {}            # pixel masks (H,W) bool (possibly resized)
        self._token_mask_cache: Dict[int, torch.Tensor] = {}    # token masks (h_llm,w_llm) bool (CPU)
        self._token_grid_cache: Dict[int, Tuple[int, int]] = {} # (h_llm,w_llm) per sample
        self._json_cache: Dict[int, Dict[str, Any]] = {}

        print(f"[ConsensusMaskDataset] Initialized with {len(self)} samples")
        print(f"  - Use precomputed masks: {self.use_precomputed_masks}")
        print(f"  - Image caching: {self.cache_images}")
        print(f"  - Precompute token masks: {self.precompute_token_masks}")
        print(f"  - max_side: {self.max_side if self.max_side > 0 else 'disabled'}")

        if self.precompute_token_masks:
            if self.model is None or self.processor is None:
                raise ValueError("model and processor must be provided when precompute_token_masks=True")
            self._precompute_all_token_masks()

    def __len__(self) -> int:
        return len(self.indices)

    # -------------------------
    # Resizing helpers
    # -------------------------
    def _resample_lanczos(self):
        # Pillow compatibility
        return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

    def _maybe_downscale_image(self, image: Image.Image) -> Image.Image:
        if self.max_side <= 0:
            return image
        w, h = image.size
        m = max(w, h)
        if m <= self.max_side:
            return image
        scale = self.max_side / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return image.resize((new_w, new_h), resample=self._resample_lanczos())

    def _resize_mask_to_image(self, mask_hw: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        mask_hw: bool array (H,W)
        returns: bool array resized to image.size (H',W') using nearest neighbor
        """
        target_w, target_h = image.size
        H, W = mask_hw.shape
        if (W, H) == (target_w, target_h):
            return mask_hw

        # NOTE: use Resampling.NEAREST if available; else Image.NEAREST
        nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)

        mask_img = Image.fromarray((mask_hw.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize((target_w, target_h), resample=nearest)
        return (np.array(mask_img) > 127).astype(bool)

    # -------------------------
    # Loading JSON/image/masks
    # -------------------------
    def _load_json(self, idx: int) -> Dict[str, Any]:
        if idx in self._json_cache:
            return self._json_cache[idx]

        json_path = self.index.get_path(idx)
        if json_path is None:
            raise ValueError(f"No consensus data found for index {idx}")

        with open(json_path, "r") as f:
            data = json.load(f)

        self._json_cache[idx] = data
        return data

    def _load_image(self, idx: int) -> Image.Image:
        if self.cache_images and idx in self._image_cache:
            return self._image_cache[idx]

        if idx >= len(self.naturalbench):
            raise IndexError(f"Index {idx} out of bounds for NaturalBench dataset")

        image = self.naturalbench[idx]["Image"]
        # Downscale early so everything downstream matches
        image = self._maybe_downscale_image(image)

        if self.cache_images:
            self._image_cache[idx] = image
        return image

    def _load_mask_from_png(self, idx: int, target_image: Optional[Image.Image] = None) -> np.ndarray:
        if self.cache_masks and idx in self._mask_cache:
            return self._mask_cache[idx]

        mask_path = self.consensus_dir / f"mask_{idx}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask PNG not found: {mask_path}")

        mask_img = Image.open(mask_path).convert("L")
        mask_array = (np.array(mask_img) > 127).astype(bool)

        # Force mask to match (possibly downscaled) image size
        if target_image is None:
            target_image = self._load_image(idx)
        mask_array = self._resize_mask_to_image(mask_array, target_image)

        if self.cache_masks:
            self._mask_cache[idx] = mask_array
        return mask_array

    def _load_mask_from_original(self, idx: int, consensus_mask_indices: List[int]) -> np.ndarray:
        if self.masks_dir is None:
            raise ValueError("masks_dir must be set to load original masks")

        all_masks = load_masks_for_index(str(self.masks_dir), idx)
        if all_masks is None:
            raise FileNotFoundError(f"Original masks not found for index {idx}")

        h, w = all_masks.shape[1], all_masks.shape[2]
        consensus_mask = np.zeros((h, w), dtype=bool)

        for mi in consensus_mask_indices:
            if 0 <= mi < all_masks.shape[0]:
                consensus_mask |= all_masks[mi]

        return consensus_mask

    # -------------------------
    # Processor inputs for grid
    # -------------------------
    def _build_inputs_for_grid(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Build processor inputs (CPU tensors) for a given image, so we can read inputs["image_grid_thw"].
        No model forward needed.
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.grid_prompt_text},
            ]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs

    # -------------------------
    # Precompute token masks
    # -------------------------
    def _precompute_all_token_masks(self):
        """
        Precompute token mask per sample, because token grid differs per image (esp after resize).
        Stores:
          self._token_mask_cache[idx] : CPU bool tensor (h_llm,w_llm)
          self._token_grid_cache[idx] : (h_llm,w_llm)
        """
        print("[ConsensusMaskDataset] Precomputing token masks (per-sample grids)...")

        mask_files = sorted(self.consensus_dir.glob("mask_*.png"))
        print(f"[ConsensusMaskDataset] Found {len(mask_files)} mask files")

        converted = 0
        skipped = 0

        for mask_path in mask_files:
            m = re.search(r"mask_(\d+)\.png$", mask_path.name)
            if not m:
                continue
            idx = int(m.group(1))

            # Only keep indices that exist in our consensus JSON index
            if idx not in self.index:
                continue

            try:
                # Load (possibly downscaled) image
                image = self._load_image(idx)

                # Build inputs from resized image (grid depends on this)
                inputs = self._build_inputs_for_grid(image)

                # Infer token grid
                t, h_llm, w_llm = llm_image_token_grid_from_inputs(inputs, self.model, self.processor)
                if t != 1:
                    raise RuntimeError(f"Expected t=1 for static image; got t={t}")

                # Load pixel mask and force match to resized image size
                mask_img = Image.open(mask_path).convert("L")
                pixel_mask = (np.array(mask_img) > 127).astype(bool)
                pixel_mask = self._resize_mask_to_image(pixel_mask, image)

                # Convert pixel mask -> LLM token mask
                tok = pixel_mask_to_llm_token_mask_from_inputs(
                    pixel_mask_hw=pixel_mask,
                    inputs=inputs,
                    model=self.model,
                    processor=self.processor,
                    image_size_wh=image.size,
                )  # np.bool_ (h_llm,w_llm)

                tok_t = torch.from_numpy(tok).to(dtype=torch.bool)  # keep CPU
                verify_token_mask_matches_vision_span(tok_t, inputs, self.model)

                self._token_mask_cache[idx] = tok_t
                self._token_grid_cache[idx] = (h_llm, w_llm)
                converted += 1

                if self.verbose_every and (converted % self.verbose_every == 0):
                    print(
                        f"  Converted {converted} masks... "
                        f"(latest idx={idx}, grid={h_llm}x{w_llm}, img={image.size})"
                    )

            except Exception as e:
                skipped += 1
                print(f"  Warning: Failed to convert mask_{idx}.png: {e}")

        print(f"[ConsensusMaskDataset] Precomputed {converted} token masks (skipped {skipped})")
        if converted > 0:
            total_elems = sum(int(m.numel()) for m in self._token_mask_cache.values())
            approx_mb = total_elems / (1024**2)  # ~1 byte/elem ballpark for bool
            print(f"  Approx token-mask storage: ~{approx_mb:.2f} MB (bool, CPU)")

    # -------------------------
    # Get item
    # -------------------------
    def __getitem__(self, dataset_idx: int) -> Dict[str, Any]:
        image_idx = self.indices[dataset_idx]

        data = self._load_json(image_idx)
        image = self._load_image(image_idx)  # already downscaled if needed

        if self.image_transform is not None:
            # WARNING: if this changes size, it breaks alignment with precomputed token masks
            image = self.image_transform(image)

        result: Dict[str, Any] = {
            "image": image,
            "question": data["question"],
            "answer_choices": data["answer_choices"],
            "gold_letter": data["gold_letter"],
            "image_idx": image_idx,
            "metadata": {
                "consensus_mask_indices": data.get("consensus_mask_indices", []),
                "gold_answer": data.get("gold_answer", ""),
                "spurious_attribute": data.get("spurious_attribute", "unknown"),
            },
        }

        if self.precompute_token_masks:
            if image_idx not in self._token_mask_cache:
                raise RuntimeError(
                    f"Token mask not found for index {image_idx}. "
                    f"It should have been precomputed at initialization."
                )
            result["token_mask"] = self._token_mask_cache[image_idx]    # CPU bool (h_llm,w_llm)
            result["token_grid_hw"] = self._token_grid_cache[image_idx] # (h_llm,w_llm)
        else:
            # Legacy: return pixel mask (still resized to match downscaled image)
            if self.use_precomputed_masks:
                pixel_mask = self._load_mask_from_png(image_idx, target_image=image)
            else:
                pixel_mask = self._load_mask_from_original(image_idx, data["consensus_mask_indices"])
                pixel_mask = self._resize_mask_to_image(pixel_mask, image)
            result["pixel_mask"] = pixel_mask

        return result


def consensus_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate that supports variable-size token masks by keeping them as lists.
    Images are kept as a list of PIL images.
    """
    out: Dict[str, Any] = {}

    out["image"] = [b["image"] for b in batch]
    out["question"] = [b["question"] for b in batch]
    out["answer_choices"] = [b["answer_choices"] for b in batch]
    out["gold_letter"] = [b["gold_letter"] for b in batch]
    out["image_idx"] = [b["image_idx"] for b in batch]
    out["metadata"] = [b["metadata"] for b in batch]

    if "token_mask" in batch[0]:
        out["token_mask"] = [b["token_mask"] for b in batch]        # list[BoolTensor(h,w)]
        out["token_grid_hw"] = [b["token_grid_hw"] for b in batch]  # list[(h,w)]
    if "pixel_mask" in batch[0]:
        out["pixel_mask"] = [b["pixel_mask"] for b in batch]

    return out


def create_dataloader(
    consensus_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = ConsensusMaskDataset(consensus_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=consensus_collate_fn,
    )