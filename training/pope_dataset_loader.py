#!/usr/bin/env python3
"""
POPE Dataset Loader for Evaluation

Loads the POPE (Polling-based Object Probing Evaluation) benchmark dataset
from HuggingFace (`lmms-lab/POPE`).  Each sample is a yes/no object
hallucination question with a category (adversarial / popular / random).

Provides the same `__len__` / `__getitem__` interface as MMVPDataset so it
can be used as a drop-in replacement in the evaluate_pope.py pipeline.

Dataset: https://huggingface.co/datasets/lmms-lab/POPE
"""

from typing import Dict, List, Any, Optional

from PIL import Image
import datasets


DEFAULT_POPE_CACHE_DIR = "/mnt/shared/shared_hf_home/datasets"
DEFAULT_MAX_IMAGE_SIDE = 1024
ANSWER_CHOICES = ["Yes", "No"]


def _resample_lanczos():
    """Pillow compatibility for LANCZOS resampling."""
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def maybe_downscale_image(
    image: Image.Image,
    max_side: int = DEFAULT_MAX_IMAGE_SIDE,
) -> Image.Image:
    if max_side <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=_resample_lanczos())


class POPEDataset:
    """
    POPE Benchmark Dataset for VLM evaluation.

    Each sample (via ``__getitem__``) returns a dict with:
        - image:          PIL Image (RGB, optionally downscaled)
        - question:       Question text  (e.g. "Is there a dog in the image?")
        - answer_choices: ["Yes", "No"]
        - gold_letter:    "A" (Yes) or "B" (No)
        - gold_answer:    "Yes" or "No"
        - index:          Integer dataset index
        - category:       One of "adversarial", "popular", "random"
        - image_source:   Source identifier for the image
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_POPE_CACHE_DIR,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    ):
        self.cache_dir = cache_dir
        self.max_image_side = max_image_side

        print(f"[POPEDataset] Loading lmms-lab/POPE (split=test, cache={cache_dir})...")
        self._hf_ds = datasets.load_dataset(
            "lmms-lab/POPE",
            cache_dir=cache_dir,
            split="test",
        )
        print(f"[POPEDataset] Loaded {len(self._hf_ds)} samples")

    def __len__(self) -> int:
        return len(self._hf_ds)

    def __getitem__(self, dataset_idx: int) -> Dict[str, Any]:
        dp = self._hf_ds[dataset_idx]

        img = dp["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = maybe_downscale_image(img, self.max_image_side)

        ans_raw = dp["answer"].strip().lower()
        if ans_raw == "yes":
            gold_letter = "A"
            gold_answer = "Yes"
        else:
            gold_letter = "B"
            gold_answer = "No"

        return {
            "image": img,
            "question": dp["question"],
            "answer_choices": list(ANSWER_CHOICES),
            "gold_letter": gold_letter,
            "gold_answer": gold_answer,
            "index": dataset_idx,
            "category": dp["category"],
            "image_source": dp["image_source"],
        }


def main():
    """Quick sanity check: load the dataset and print a few samples."""
    ds = POPEDataset()
    print(f"\nTotal samples: {len(ds)}")

    for i in range(min(5, len(ds))):
        s = ds[i]
        print(
            f"  [{i}] category={s['category']:12s}  "
            f"gold={s['gold_letter']} ({s['gold_answer']:3s})  "
            f"q={s['question'][:60]}"
        )


if __name__ == "__main__":
    main()
