#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# ----------------------------
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE any CUDA-related imports
# ----------------------------
def _early_parse_devices(argv=None):
    """Parse GPU device arguments before any imports that might initialize CUDA."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vlm_device", type=int, required=True)
    p.add_argument("--inpaint_device", type=int, required=True)
    args, _ = p.parse_known_args(argv)
    return args.vlm_device, args.inpaint_device


_vlm_dev, _inp_dev = _early_parse_devices()

# Support same-GPU or multi-GPU operation
if _vlm_dev == _inp_dev:
    # Same GPU: only make that GPU visible
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{_vlm_dev}"
    _SAME_GPU = True
    print(f"[GPU config] Using single GPU {_vlm_dev} for both VLM and inpainting")
else:
    # Different GPUs: Map visible GPU 0 -> VLM, visible GPU 1 -> inpaint
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{_vlm_dev},{_inp_dev}"
    _SAME_GPU = False
    print(f"[GPU config] Using GPU {_vlm_dev} for VLM (cuda:0) and GPU {_inp_dev} for inpainting (cuda:1)")

os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Now safe to import everything else (including torch)
import re
import json
import glob
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from diffusers import StableDiffusionInpaintPipeline
import signal
import torch.distributed as dist


class NsfwRetryExhausted(RuntimeError):
    pass


def _nsfw_flags_from_pipe_output(pipe_out, n: int) -> List[bool]:
    """
    Diffusers versions differ:
      - nsfw_content_detected (newer)
      - has_nsfw_concept (older)
    Either may be bool or list[bool]. Normalize to list[bool] of length n.
    """
    for attr in ("nsfw_content_detected", "has_nsfw_concept"):
        if hasattr(pipe_out, attr):
            v = getattr(pipe_out, attr)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                flags = [bool(x) for x in v]
                if len(flags) == n:
                    return flags
                # if mismatch, fall back to conservative broadcast
                return [any(flags)] * n
            return [bool(v)] * n
    return [False] * n


def _is_black_image(img) -> bool:
    arr = np.asarray(img)
    return (arr.size > 0) and (arr.max() == 0)


def _guess_repo_id_from_models_dir(models_dir: str) -> str:
    """
    /.../models--ORG--REPO  ->  ORG/REPO
    """
    base = Path(models_dir).name
    m = re.match(r"models--([^/]+)--(.+)$", base)
    if not m:
        raise ValueError(f"Can't infer repo id from: {models_dir}")
    return f"{m.group(1)}/{m.group(2)}"

def _shared_hf_hub_cache(models_dir: str) -> str:
    """
    Your paths look like:
      /mnt/shared/shared_hf_home/models--...
      /mnt/shared/shared_hf_home/hub/models--...
    So hub cache root is:
      /mnt/shared/shared_hf_home/hub
    """
    p = Path(models_dir)
    # if we are already under .../hub/models--..., hub root is parent of models--...
    if p.parent.name == "hub":
        return str(p.parent)
    # otherwise assume sibling "hub"
    return str(p.parent / "hub")


def _cleanup_distributed():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # IO
    ap.add_argument("--patch_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="inpainting_search_result")
    ap.add_argument(
        "--questions_json",
        type=str,
        default="/mnt/arc/zhaonan2/blind_project/datasets/seed_bench/seed_bench_image_questions.json",
        help="Path to seed bench questions JSON (list of items with path, question, ground_truth, etc.).",
    )
    ap.add_argument(
        "--seed_bench_root",
        type=str,
        default="/mnt/arc/zhaonan2/blind_project/datasets/seed_bench",
        help="Root directory for seed bench; image path = seed_bench_root + item['path'].",
    )
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)

    # GPUs (required early)
    ap.add_argument("--vlm_device", type=int, required=True)
    ap.add_argument("--inpaint_device", type=int, required=True)

    # Search
    ap.add_argument("--num_patches", type=int, default=1)
    ap.add_argument("--num_samples_sc", type=int, default=15)
    ap.add_argument("--min_sc_drop", type=float, default=0.1)
    ap.add_argument(
        "--require_monotonic_improvement",
        action="store_true",
        help="Require each new patch to increase (baseline_sc_target - current_sc_target).",
    )

    # VLM / vLLM
    ap.add_argument(
        "--vlm_model_dir",
        type=str,
        required=True,
        help="Path to local HF hub folder (models--...); script will auto-resolve snapshots/<hash>/.",
    )
    ap.add_argument("--model_tag", type=str, default=None, help="Optional output tag; defaults to folder name.")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)

    # Inpainting
    ap.add_argument(
        "--inpaint_model_id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    ap.add_argument(
        "--inpaint_prompt",
        type=str,
        default="Fill the masked region naturally, removing whatever objects appear there. Preserve the rest of the image exactly.",
    )
    ap.add_argument(
        "--inpaint_negative_prompt",
        type=str,
        default="text, watermark, logo, artifacts, distortion, blur",
    )
    ap.add_argument("--inpaint_steps", type=int, default=30)
    ap.add_argument("--inpaint_guidance", type=float, default=7.5)
    ap.add_argument("--inpaint_strength", type=float, default=1.0)
    ap.add_argument("--inpaint_batch_size", type=int, default=8)

    # Optional: Qwen-generated diffusion prompts
    ap.add_argument("--use_qwen_diffusion_prompt", action="store_true")
    ap.add_argument("--qwen_prompt_temperature", type=float, default=0.2)
    ap.add_argument("--qwen_prompt_top_p", type=float, default=0.9)
    ap.add_argument("--qwen_prompt_max_tokens", type=int, default=256)

    # Optional fixed resolution casting
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--no_resize", action="store_true")

    return ap.parse_args()


# ----------------------------
# Model path resolver
# ----------------------------
def resolve_hf_snapshot_dir(model_dir: str, allow_missing: bool = False) -> Optional[str]:
    """
    Accepts either:
      - a resolved snapshot directory containing config.json
      - a HF hub models--ORG--NAME directory containing snapshots/<hash>/
    Returns a directory that contains config.json, or None if allow_missing=True and not found.
    """
    p = Path(model_dir)
    if (p / "config.json").exists():
        return str(p)

    snaps = sorted(glob.glob(str(p / "snapshots" / "*")), key=lambda x: os.path.getmtime(x))
    if not snaps:
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"Could not find snapshots under: {model_dir}\n"
            f"Expected: {model_dir}/snapshots/<hash>/config.json"
        )
    snap = Path(snaps[-1])
    if not (snap / "config.json").exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Snapshot missing config.json: {snap}")
    return str(snap)


def default_model_tag(model_dir: str) -> str:
    base = Path(model_dir).name
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base[:120]


# ----------------------------
# Dataset (seed bench)
# ----------------------------
def seed_bench_item_to_dict(item: Dict[str, Any], seed_bench_root: str) -> Dict[str, Any]:
    """
    Map a seed bench JSON item to the same data_dict shape expected by the rest of the pipeline.
    item has: path, question, ground_truth, question_id, question_type_id.
    """
    img_path = os.path.join(seed_bench_root, item["path"])
    image = Image.open(img_path).convert("RGB")

    question_raw = item["question"]
    ground_truth = str(item.get("ground_truth", "")).strip()
    # ground_truth is e.g. "(A)" -> gold_letter "A"
    gold_letter = ""
    if len(ground_truth) >= 1:
        c = ground_truth.lstrip("(").strip()[0].upper()
        if c in "ABCDE":
            gold_letter = c

    # Parse answer choices: (A) One. (B) Two. (C) Three. (D) Four.
    answer_choices: List[str] = []
    option_pattern = re.compile(r"\(([A-E])\)\s*(.+?)(?=\s*\([A-E]\)|$)", re.DOTALL)
    for m in option_pattern.finditer(question_raw):
        choice_text = m.group(2).strip().rstrip(".").strip()
        answer_choices.append(choice_text)

    if not answer_choices:
        answer_choices = ["Option A", "Option B", "Option C", "Option D"][: max(2, ord(gold_letter) - ord("A") + 1) if gold_letter else 2]

    gold_answer = ""
    if gold_letter:
        idx = ord(gold_letter) - ord("A")
        gold_answer = answer_choices[idx] if 0 <= idx < len(answer_choices) else str(item.get("ground_truth", ""))

    return {
        "image": image,
        "question": question_raw,
        "answer_choices": answer_choices,
        "gold_answer": gold_answer,
        "gold_letter": gold_letter,
        "spurious_attribute": item.get("question_type_id", ""),
    }


# ----------------------------
# Resize/pad helpers
# ----------------------------
def maybe_downscale_image(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Downscale image if any dimension exceeds max_side, preserving aspect ratio."""
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Use LANCZOS/Resampling.LANCZOS for high-quality downscaling
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
    return img.resize((new_w, new_h), resample=resample)


def resize_pad_to_square(img: Image.Image, size: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_rs = img.resize((new_w, new_h), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (240, 240, 240))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(img_rs, (pad_left, pad_top))
    return canvas, (pad_left, pad_top, new_w, new_h)


def resize_mask_to_image(mask_hw: np.ndarray, target_image: Image.Image) -> np.ndarray:
    """Resize mask to match target image size using nearest neighbor."""
    target_w, target_h = target_image.size
    H, W = mask_hw.shape
    if (W, H) == (target_w, target_h):
        return mask_hw
    
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
    mask_img = Image.fromarray((mask_hw.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.resize((target_w, target_h), resample=nearest)
    return (np.array(mask_img) > 127).astype(bool)


def resize_pad_mask(mask_bool: np.ndarray, orig_size: Tuple[int, int], size: int, pad_info: Tuple[int, int, int, int]) -> np.ndarray:
    orig_w, orig_h = orig_size
    if mask_bool.shape != (orig_h, orig_w):
        raise ValueError(f"Mask shape {mask_bool.shape} != orig image (H,W)=({orig_h},{orig_w}).")

    pad_left, pad_top, new_w, new_h = pad_info
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    mask_rs = mask_img.resize((new_w, new_h), resample=Image.NEAREST)

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(mask_rs, (pad_left, pad_top))
    return (np.array(canvas) > 0)


# ----------------------------
# Prompting + parsing
# ----------------------------
def make_text_prompt(question: str, answer_choices: List[str]) -> str:
    prompt = (
        "Given the following question, and answer choices, do the following:\n"
        "1) Think step by step, first identify what you can see in the image that is relevant to the question.\n"
        "2) Based on this information, reason about each answer choice and how likely it is to be correct.\n"
        "3) Finally, select the index of the single best answer choice.\n\n"
        f"Question: {question}\n"
        "Answer Choices:\n"
    )
    options = ["A", "B", "C", "D", "E"]
    for i, choice in enumerate(answer_choices):
        prompt += f"{options[i]}: {choice}\n"
    prompt += "\nPlease provide the final answer in this JSON format: {'final_answer_letter': <letter>}\n"
    return prompt


def parse_qwen_output(text: str) -> Optional[str]:
    m = re.search(r"\{.*?final_answer_letter.*?\}", text, re.DOTALL)
    if m:
        json_str = m.group(0)
        json_str_norm = json_str.replace("'", '"')
        json_str_norm = re.sub(r",\s*}", "}", json_str_norm)
        try:
            obj = json.loads(json_str_norm)
            if "final_answer_letter" in obj:
                value = str(obj["final_answer_letter"]).strip().upper()
                if len(value) == 1 and value in "ABCDE":
                    return value
                return None
        except Exception:
            pass

    m2 = re.search(r"final_answer_letter\s*[:=]\s*['\"]?([A-Ea-e])['\"]?", text)
    if m2:
        return m2.group(1).upper()

    m3 = re.search(r"^\s*([A-E])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()

    return None


def sc_wrt_letter(counts: Dict[str, int], num_extracted: int, letter: Optional[str]) -> Optional[float]:
    if num_extracted <= 0 or letter is None:
        return None
    return counts.get(letter, 0) / num_extracted


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def self_consistency_batch(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    images: List[Image.Image],
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    question = make_text_prompt(data_dict["question"], data_dict["answer_choices"])

    vllm_inputs = []
    for im in images:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": im}, {"type": "text", "text": question}],
        }]
        vllm_inputs.append(prepare_inputs_for_vllm(messages, processor))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples_sc,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    results = []
    gold_letter = data_dict["gold_letter"]
    for out in outputs:
        num_extracted = 0
        letter_counts: Dict[str, int] = {}
        for gen in out.outputs:
            parsed = parse_qwen_output(gen.text)
            if parsed is not None:
                num_extracted += 1
                letter_counts[parsed] = letter_counts.get(parsed, 0) + 1

        majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
        gold_sc = (letter_counts.get(gold_letter, 0) / num_extracted) if (num_extracted > 0 and gold_letter) else None

        results.append({
            "num_extracted": num_extracted,
            "letter_counts": letter_counts,
            "majority_letter": majority_letter,
            "gold_sc": gold_sc,
        })

    return results


# ----------------------------
# Inpainting
# ----------------------------
def load_inpaint_pipe(model_id: str):
    # Use cuda:0 if same GPU, cuda:1 if separate GPUs
    if torch.cuda.is_available():
        device = "cuda:0" if _SAME_GPU else "cuda:1"
    else:
        device = "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    return pipe, device


def mask_bool_to_pil_L(mask_bool: np.ndarray) -> Image.Image:
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def inpaint_batch(
    pipe: StableDiffusionInpaintPipeline,
    device: str,
    base_image: Image.Image,
    masks_bool: List[np.ndarray],
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    strength: float,
    batch_size: int,
    seed: int,
    max_retries: int = 3,        # retry count
    seed_step: int = 100000,     # big jump so retries differ
) -> List[Image.Image]:
    """
    Runs batched inpainting. If safety checker triggers for some outputs (or black image returned),
    retries ONLY those items up to max_retries times. If any items still fail after retries,
    raises NsfwRetryExhausted so caller can skip the whole dataset index.
    """
    out_images: List[Image.Image] = []
    if not masks_bool:
        return out_images

    use_cuda = device.startswith("cuda")

    w, h = base_image.size
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    for s in range(0, len(masks_bool), batch_size):
        chunk = masks_bool[s:s + batch_size]
        n_chunk = len(chunk)

        # results for this chunk; fill as items succeed
        chunk_results: List[Optional[Image.Image]] = [None] * n_chunk
        pending = list(range(n_chunk))

        for attempt in range(max_retries + 1):
            if not pending:
                break

            # build a sub-batch for only the still-pending items
            sub_masks = [mask_bool_to_pil_L(chunk[i]) for i in pending]
            sub_imgs = [base_image] * len(pending)

            # IMPORTANT: different generator per item (and per attempt) so retries meaningfully differ
            if use_cuda:
                gens = [
                    torch.Generator(device=device).manual_seed(
                        seed
                        + (s + i) * 17              # per-item offset
                        + attempt * seed_step        # per-attempt offset
                    )
                    for i in pending
                ]
            else:
                gens = None

            with torch.inference_mode():
                res = pipe(
                    prompt=[prompt] * len(pending),
                    negative_prompt=[negative_prompt] * len(pending),
                    image=sub_imgs,
                    mask_image=sub_masks,
                    height=h8,
                    width=w8,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    strength=strength,
                    generator=gens,
                )

            if use_cuda:
                torch.cuda.synchronize(torch.device(device))

            sub_out = res.images
            nsfw_flags = _nsfw_flags_from_pipe_output(res, n=len(pending))

            # also treat pure-black image as a failure signal
            still_bad: List[int] = []
            for j, orig_idx in enumerate(pending):
                img = sub_out[j]
                flagged = bool(nsfw_flags[j]) or _is_black_image(img)
                if not flagged:
                    chunk_results[orig_idx] = img
                else:
                    still_bad.append(orig_idx)

            if still_bad and attempt < max_retries:
                print(
                    f"[inpaint] safety triggered for {len(still_bad)}/{len(pending)} "
                    f"items in chunk starting at {s}; retry {attempt+1}/{max_retries}"
                )

            pending = still_bad

        if pending:
            # retries exhausted for some items in this chunk
            raise NsfwRetryExhausted(
                f"Safety checker/black output persisted after {max_retries} retries "
                f"for {len(pending)}/{n_chunk} items (chunk start={s})."
            )

        # chunk fully successful
        out_images.extend([im for im in chunk_results if im is not None])

    del sub_masks, sub_imgs, res
    if gens is not None:
        del gens
    return out_images


# ----------------------------
# Mask IO (seed bench: stem-based)
# ----------------------------
def load_masks_for_stem(patch_dir: str, stem: str) -> Optional[np.ndarray]:
    mask_path = os.path.join(patch_dir, f"{stem}_masks.npz")
    if not os.path.exists(mask_path):
        return None
    data = np.load(mask_path)
    return data["masks"]


# ----------------------------
# vLLM init
# ----------------------------
def _filter_kwargs_for_llm(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(LLM.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def _patch_vllm_vit_attn_respect_override() -> None:
    """
    Patch vLLM so the vision encoder respects mm_encoder_attn_backend for Qwen 2.5 VL.

    vLLM's maybe_get_vit_flash_attn_backend() (in vllm/attention/layer.py) unconditionally
    upgrades any non-FLASH backend to FLASH_ATTN on CUDA when flash_attn is available,
    ignoring the user's mm_encoder_attn_backend. Qwen 2.5 VL's ViT has head_dim=80, which
    Flash Attention does not support (headdim must be a multiple of 32), so we must
    force TORCH_SDPA. This patch adds "attn_backend_override is None" to the condition
    so an explicit override is not overwritten. The patch is applied on disk so the
    vLLM worker subprocess also gets the fix.
    """
    import vllm.attention.layer as layer_mod  # noqa: PLC0415
    path = getattr(layer_mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Already patched if our new condition is present
    if "attn_backend_override is None\n            and attn_backend != AttentionBackendEnum.FLASH_ATTN" in content:
        return
    old = """    elif current_platform.is_cuda():
        if (
            attn_backend != AttentionBackendEnum.FLASH_ATTN
            and check_upstream_fa_availability(torch.get_default_dtype())
        ):
            attn_backend = AttentionBackendEnum.FLASH_ATTN
            use_upstream_fa = True"""
    new = """    elif current_platform.is_cuda():
        if (
            attn_backend_override is None
            and attn_backend != AttentionBackendEnum.FLASH_ATTN
            and check_upstream_fa_availability(torch.get_default_dtype())
        ):
            attn_backend = AttentionBackendEnum.FLASH_ATTN
            use_upstream_fa = True"""
    if old not in content:
        print("[vLLM] WARNING: Could not patch vllm/attention/layer.py (source changed?). Vision encoder may still use Flash Attn.")
        return
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[vLLM] Patched vllm/attention/layer.py so mm_encoder_attn_backend is respected (avoid Flash Attn for head_dim 80).")


def init_vllm_engine(model_path: str, args) -> LLM:
    """
    Robust vLLM init that:
      - Uses local HF cache layout you have (models--ORG--REPO + shared hub cache)
      - Forces Qwen2.5-VL-72B to use fp8 weight quant + fp8 KV + calculate_kv_scales=True
      - Uses enforce_eager=True for Qwen 2.5 VL (standard attention, avoids Flash Attn head_dim issues)
      - Ensures max_model_len is actually passed (no signature-based filtering)
      - Drops only truly-unsupported kwargs if this vLLM build rejects them
    """
    # Use local path when we have a resolved snapshot; otherwise repo_id for Hub download.
    repo_id = _guess_repo_id_from_models_dir(args.vlm_model_dir)
    vlm_model = model_path if (os.path.isabs(model_path) and os.path.isdir(model_path)) else repo_id
    rid = model_path.lower() + " " + repo_id.lower()
    is_qwen25 = ("qwen2.5" in rid) or ("qwen2_5" in rid) or ("qwen2-5" in rid) or ("qwen25" in rid)
    use_eager = is_qwen25
    hub_cache = _shared_hf_hub_cache(args.vlm_model_dir)

    # Qwen 2.5 VL has vision encoder head_dim 80 (not divisible by 32); Flash Attention fails.
    if use_eager:
        print("[vLLM] Using enforce_eager=True for Qwen 2.5 VL (standard attention)")
    else:
        print("[vLLM] Using enforce_eager=False for Qwen 2.5 VL (flash attention)")

    llm_kwargs: Dict[str, Any] = dict(
        model=vlm_model,
        tokenizer=vlm_model,

        # Offline/shared-cache hints (some vLLM versions accept one or both)
        download_dir=hub_cache,
        hf_cache_dir=hub_cache,

        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,   # IMPORTANT: must not be dropped
        seed=args.seed,
        enforce_eager=use_eager,
        dtype="auto",

        # harmless if ignored by this vLLM build
        trust_remote_code=True,
        disable_log_stats=True,
    )

    already_quantized = "fp8" in rid and ("redhat" in rid or "neuralmagic" in rid or "compressed" in rid or "dynamic" in rid)

    if is_qwen25:
        _patch_vllm_vit_attn_respect_override()
        llm_kwargs["mm_encoder_attn_backend"] = "TORCH_SDPA"
        print("[vLLM] Using mm_encoder_attn_backend=TORCH_SDPA for Qwen 2.5 VL (vision encoder)")

        if already_quantized:
            print("[vLLM] Pre-quantized FP8 checkpoint detected — skipping on-the-fly quantization")
            llm_kwargs["kv_cache_dtype"] = "fp8"
        else:
            llm_kwargs["quantization"] = "fp8"
            llm_kwargs["kv_cache_dtype"] = "fp8"
            llm_kwargs["calculate_kv_scales"] = True
            print("[vLLM] Applying on-the-fly FP8 quantization for unquantized Qwen 2.5 checkpoint")

    # DO NOT signature-filter. Some vLLM versions hide real engine args from LLM.__init__ signature.
    # Instead, try and only strip unsupported kwargs if vLLM throws "unexpected keyword argument".
    while True:
        try:
            print(f"[vLLM] init kwargs: {llm_kwargs}")
            return LLM(**llm_kwargs)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m:
                raise
            bad = m.group(1)
            print(f"[vLLM] WARNING: dropping unsupported kwarg: {bad}")
            llm_kwargs.pop(bad, None)
            if not llm_kwargs:
                raise


# ----------------------------
# Baseline-target greedy search
# ----------------------------
def find_patches_baseline_target(
    llm: LLM,
    processor: AutoProcessor,
    inpaint_pipe: StableDiffusionInpaintPipeline,
    inpaint_device: str,
    data_dict: Dict[str, Any],
    masks: np.ndarray,
    args,
) -> Dict[str, Any]:
    original_image = data_dict["image"]
    w0, h0 = original_image.size

    if masks.dtype != bool:
        masks = masks.astype(bool)
    N, H, W = masks.shape
    assert (H, W) == (h0, w0), "Mask size must match (possibly resized) image size."

    # (1) Baseline on unperturbed image (to define FIXED target); retry once if invalid
    gold_letter = data_dict["gold_letter"]
    for baseline_attempt in range(2):
        baseline_res = self_consistency_batch(
            llm, processor, data_dict, [original_image],
            args.num_samples_sc, args.temperature, args.top_p
        )[0]
        baseline_num = baseline_res["num_extracted"]
        baseline_counts = baseline_res["letter_counts"]
        target_letter = baseline_res["majority_letter"]  # FIXED target
        baseline_sc_target = sc_wrt_letter(baseline_counts, baseline_num, target_letter)
        baseline_sc_gold = baseline_res["gold_sc"]
        if target_letter is not None and baseline_sc_target is not None:
            break
        if baseline_attempt == 0:
            print("[baseline] No valid target/SC (parsing failed); retrying once...")
        else:
            print("[baseline] No valid baseline after retry; skipping sample.")
            return {
                "skipped": True,
                "skip_reason": "no_valid_baseline",
                "selected_mask_indices": [],
                "final_image": original_image,
                "baseline_majority_letter": None,
                "final_majority_letter": None,
                "baseline_num_extracted": baseline_num,
                "final_num_extracted": baseline_num,
                "baseline_sc_target": None,
                "final_sc_target": None,
                "baseline_sc_gold": baseline_sc_gold,
                "final_sc_gold": baseline_sc_gold,
            }

    print(f"[baseline] target_letter={target_letter} baseline_sc_target={baseline_sc_target} baseline_sc_gold={baseline_sc_gold}")

    global_mask = np.zeros((H, W), dtype=bool)
    selected_indices: List[int] = []

    # Greedy selection
    for step_i in range(args.num_patches):
        print(f"\n--- Step {step_i+1}/{args.num_patches} --- FIXED target={target_letter}")

        # Current image = inpaint(original, global_mask)
        if global_mask.any():
            cur_img = inpaint_batch(
                pipe=inpaint_pipe,
                device=inpaint_device,
                base_image=original_image,
                masks_bool=[global_mask],
                prompt=args.inpaint_prompt,
                negative_prompt=args.inpaint_negative_prompt,
                num_steps=args.inpaint_steps,
                guidance=args.inpaint_guidance,
                strength=args.inpaint_strength,
                batch_size=1,
                seed=args.seed + 1000 + step_i,
            )[0]
        else:
            cur_img = original_image

        cur_res = self_consistency_batch(
            llm, processor, data_dict, [cur_img],
            args.num_samples_sc, args.temperature, args.top_p
        )[0]
        cur_num = cur_res["num_extracted"]
        cur_counts = cur_res["letter_counts"]
        cur_sc_target = sc_wrt_letter(cur_counts, cur_num, target_letter)

        if cur_sc_target is None or target_letter is None:
            print("No valid SC current/target (parsing failed); stopping.")
            break

        # For logging only: cumulative drop from baseline (not used for selection anymore)
        cum_drop_from_baseline = (baseline_sc_target - cur_sc_target) if (baseline_sc_target is not None) else None
        if cum_drop_from_baseline is None:
            print(f"[current] sc_target={cur_sc_target:.4f} majority(log)={cur_res['majority_letter']}")
        else:
            print(f"[current] sc_target={cur_sc_target:.4f} cum_drop_from_baseline={cum_drop_from_baseline:.4f} majority(log)={cur_res['majority_letter']}")

        cand_indices = [m_idx for m_idx in range(N) if m_idx not in selected_indices]
        if not cand_indices:
            print("No remaining candidates; stopping.")
            break

        cand_masks = [global_mask | masks[m_idx] for m_idx in cand_indices]

        # Inpaint all candidates
        cand_images = inpaint_batch(
            pipe=inpaint_pipe,
            device=inpaint_device,
            base_image=original_image,
            masks_bool=cand_masks,
            prompt=args.inpaint_prompt,
            negative_prompt=args.inpaint_negative_prompt,
            num_steps=args.inpaint_steps,
            guidance=args.inpaint_guidance,
            strength=args.inpaint_strength,
            batch_size=args.inpaint_batch_size,
            seed=args.seed + 2000 + step_i,
        )

        # SC on candidate images
        cand_sc_results = self_consistency_batch(
            llm, processor, data_dict, cand_images,
            args.num_samples_sc, args.temperature, args.top_p
        )

        # Selection now uses *incremental* drop vs CURRENT sc_target
        best_idx = None
        best_step_drop = float("-inf")   # cur_sc_target - cand_sc_target
        best_sc_after = None

        for m_idx, res in zip(cand_indices, cand_sc_results):
            cand_num = res["num_extracted"]
            cand_counts = res["letter_counts"]
            cand_sc_target = sc_wrt_letter(cand_counts, cand_num, target_letter)
            if cand_sc_target is None:
                continue

            step_drop = cur_sc_target - cand_sc_target
            print(f"  cand mask {m_idx}: sc_target_after={cand_sc_target:.4f} step_drop_from_current={step_drop:.4f}")

            # Must actually decrease vs current by at least min_sc_drop
            if step_drop < args.min_sc_drop:
                continue

            # If enabled, enforce strictly monotonic decrease (redundant with min_sc_drop>0, but safe)
            if args.require_monotonic_improvement and step_drop <= 1e-9:
                continue

            if step_drop > best_step_drop:
                best_step_drop = step_drop
                best_idx = m_idx
                best_sc_after = cand_sc_target

        if best_idx is None:
            print(f"No candidate achieved step_drop_from_current >= {args.min_sc_drop}. Stop.")
            break

        print(f"[select] idx={best_idx} sc_target_after={best_sc_after:.4f} step_drop_from_current={best_step_drop:.4f}")
        global_mask |= masks[best_idx]
        selected_indices.append(best_idx)

    # Final image
    if global_mask.any():
        final_img = inpaint_batch(
            pipe=inpaint_pipe,
            device=inpaint_device,
            base_image=original_image,
            masks_bool=[global_mask],
            prompt=args.inpaint_prompt,
            negative_prompt=args.inpaint_negative_prompt,
            num_steps=args.inpaint_steps,
            guidance=args.inpaint_guidance,
            strength=args.inpaint_strength,
            batch_size=1,
            seed=args.seed + 9999,
        )[0]
    else:
        final_img = original_image

    final_res = self_consistency_batch(
        llm, processor, data_dict, [final_img],
        args.num_samples_sc, args.temperature, args.top_p
    )[0]
    final_num = final_res["num_extracted"]
    final_counts = final_res["letter_counts"]
    final_majority = final_res["majority_letter"]
    final_sc_target = sc_wrt_letter(final_counts, final_num, target_letter)
    final_sc_gold = sc_wrt_letter(final_counts, final_num, gold_letter)

    try:
        del cand_images, cand_masks, cand_sc_results
    except NameError:
        pass  # loop broke before first assignment (e.g. parsing failed)

    return {
        "selected_mask_indices": selected_indices,
        "final_image": final_img,
        "baseline_majority_letter": target_letter,
        "final_majority_letter": final_majority,
        "baseline_num_extracted": baseline_num,
        "final_num_extracted": final_num,
        "baseline_sc_target": baseline_sc_target,
        "final_sc_target": final_sc_target,
        "baseline_sc_gold": baseline_sc_gold,
        "final_sc_gold": final_sc_gold,
    }


# ----------------------------
# Visualization
# ----------------------------
def create_comparison_visualization(
    original_image: Image.Image,
    perturbed_image: Image.Image,
    question: str,
    gold_answer: str,
    answer_choices: List[str],
    baseline_majority_letter: Optional[str],
    final_majority_letter: Optional[str],
    baseline_sc_target: Optional[float],
    baseline_sc_gold: Optional[float],
    final_sc_target: Optional[float],
    final_sc_gold: Optional[float],
    output_path: str,
):
    import textwrap

    def _load_font(sz: int):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
        except Exception:
            return ImageFont.load_default()

    def _letter_to_text(letter: Optional[str]) -> str:
        if not letter or len(letter) != 1:
            return "N/A"
        idx = ord(letter.upper()) - ord("A")
        return answer_choices[idx] if 0 <= idx < len(answer_choices) else "N/A"

    def _fmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.4f}"

    def _resize_to_fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = im.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        return im if (nw, nh) == (w, h) else im.resize((nw, nh), resample=Image.BICUBIC)

    def _measure_multiline(draw, lines, font, line_gap=4):
        heights = []
        max_w = 0
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_w = max(max_w, w)
            heights.append(h)
        total_h = sum(heights) + max(0, len(lines) - 1) * line_gap
        return max_w, total_h

    margin = 24
    gutter = 24
    header_gap = 10
    line_gap = 4
    MAX_COL_W = 820
    MAX_IMG_H = 520

    font_title = _load_font(22)
    font_body = _load_font(16)
    font_small = _load_font(14)

    orig_rs = _resize_to_fit(original_image, MAX_COL_W, MAX_IMG_H)
    pert_rs = _resize_to_fit(perturbed_image, MAX_COL_W, MAX_IMG_H)
    col_w = max(orig_rs.size[0], pert_rs.size[0], 500)

    baseline_answer_text = _letter_to_text(baseline_majority_letter)
    final_answer_text = _letter_to_text(final_majority_letter)

    left_caption = [
        "Pre (original image)",
        f"Majority answer: {baseline_answer_text}",
        f"SC wrt TARGET (baseline): {_fmt(baseline_sc_target)}",
        f"SC wrt gold: {_fmt(baseline_sc_gold)}",
    ]
    right_caption = [
        "Post (inpainted with selected patches)",
        f"Majority answer: {final_answer_text}",
        f"SC wrt TARGET (baseline): {_fmt(final_sc_target)}",
        f"SC wrt gold: {_fmt(final_sc_gold)}",
    ]

    header_lines = ["Question:"]
    header_lines += textwrap.wrap(question, width=90) if question else ["(missing)"]
    header_lines.append(f"Gold Answer: {gold_answer}")

    scratch = Image.new("RGB", (10, 10), "white")
    d0 = ImageDraw.Draw(scratch)
    _, header_h = _measure_multiline(d0, header_lines, font_body, line_gap=line_gap)
    _, left_cap_h = _measure_multiline(d0, left_caption, font_small, line_gap=line_gap)
    _, right_cap_h = _measure_multiline(d0, right_caption, font_small, line_gap=line_gap)
    cap_h = max(left_cap_h, right_cap_h)

    canvas_w = margin * 2 + col_w * 2 + gutter
    canvas_h = (
        margin + header_h + header_gap
        + max(orig_rs.size[1], pert_rs.size[1])
        + header_gap + cap_h + margin
    )

    combined = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(combined)

    x = margin
    y = margin
    draw.text((x, y), header_lines[0], fill="black", font=font_title)
    y += (draw.textbbox((0, 0), header_lines[0], font=font_title)[3] - draw.textbbox((0, 0), header_lines[0], font=font_title)[1]) + 6

    for ln in header_lines[1:]:
        draw.text((x, y), ln, fill="black", font=font_body)
        bbox = draw.textbbox((0, 0), ln, font=font_body)
        y += (bbox[3] - bbox[1]) + line_gap

    y += header_gap

    left_x = margin
    right_x = margin + col_w + gutter
    img_y = y

    ox = left_x + (col_w - orig_rs.size[0]) // 2
    px = right_x + (col_w - pert_rs.size[0]) // 2
    combined.paste(orig_rs, (ox, img_y))
    combined.paste(pert_rs, (px, img_y))

    y = img_y + max(orig_rs.size[1], pert_rs.size[1]) + header_gap

    def _draw_caption(col_x: int, lines: List[str]):
        yy = y
        for ln in lines:
            draw.text((col_x, yy), ln, fill="black", font=font_small)
            bbox = draw.textbbox((0, 0), ln, font=font_small)
            yy += (bbox[3] - bbox[1]) + line_gap

    _draw_caption(left_x, left_caption)
    _draw_caption(right_x, right_caption)

    combined.save(output_path)
    print(f"[viz] saved {output_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # Try to resolve local snapshot; if not found, fall back to downloading from HF Hub
    model_path = resolve_hf_snapshot_dir(args.vlm_model_dir, allow_missing=True)
    repo_id = _guess_repo_id_from_models_dir(args.vlm_model_dir)
    hub_cache = _shared_hf_hub_cache(args.vlm_model_dir)
    
    use_local = model_path is not None
    if not use_local:
        print(f"[model] local snapshot not found at {args.vlm_model_dir}")
        print(f"[model] will download from HuggingFace Hub: {repo_id}")
        model_path = repo_id  # vLLM will use repo_id to download
    
    model_tag = args.model_tag or default_model_tag(args.vlm_model_dir)

    out_dir = os.path.join(args.output_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    expected_gpus = 1 if _SAME_GPU else 2
    print(f"torch sees {torch.cuda.device_count()} GPUs (expected {expected_gpus}).")
    if torch.cuda.is_available():
        print(f"vLLM GPU (cuda:0) name: {torch.cuda.get_device_name(0)}")
        if _SAME_GPU:
            print(f"inpaint GPU (cuda:0) name: {torch.cuda.get_device_name(0)} [same as VLM]")
        else:
            print(f"inpaint GPU (cuda:1) name: {torch.cuda.get_device_name(1)}")

    if use_local:
        print(f"[model] resolved snapshot: {model_path}")
        print(f"[processor] loading from: {model_path}")
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        except OSError as e:
            print(f"[processor] snapshot missing image processor config; falling back to repo_id={repo_id} using cache_dir={hub_cache}")
            processor = AutoProcessor.from_pretrained(
                repo_id,
                trust_remote_code=True,
                local_files_only=False,  # Allow download
                cache_dir=hub_cache,
            )
    else:
        print(f"[processor] downloading from HuggingFace Hub: {repo_id}")
        processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=True,
            local_files_only=False,  # Allow download
            cache_dir=hub_cache,
        )

    print("[vLLM] initializing engine (cuda:0)...")
    llm = init_vllm_engine(model_path, args)

    print("[inpaint] initializing pipeline (cuda:1)...")
    inpaint_pipe, inpaint_device = load_inpaint_pipe(args.inpaint_model_id)
    print(f"[inpaint] device = {inpaint_device}")

    # Discover which stems have masks in patch_dir
    mask_files = [f for f in os.listdir(args.patch_dir) if f.endswith("_masks.npz")]
    available_stems = sorted({f.rsplit("_masks.npz", 1)[0] for f in mask_files})
    print(f"[masks] found {len(available_stems)} stems with masks in {args.patch_dir}")

    # Load questions JSON and build stem -> list of items lookup
    print("[data] loading seed bench questions...")
    with open(args.questions_json, "r") as f:
        all_items = json.load(f)
    stem_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in all_items:
        s = Path(item["path"]).stem
        stem_to_items.setdefault(s, []).append(item)

    # Build flat work list: one (stem, item) per image (first question only)
    work_list: List[Tuple[str, Dict[str, Any]]] = []
    for stem in available_stems:
        items_for_stem = stem_to_items.get(stem, [])
        if items_for_stem:
            work_list.append((stem, items_for_stem[0]))
    n_total = len(work_list)
    print(f"[data] {n_total} (stem, question) pairs to consider")

    start_idx = max(0, args.start_idx)
    end_idx = (n_total - 1) if args.end_idx < 0 else min(args.end_idx, n_total - 1)
    if start_idx > end_idx:
        raise SystemExit(f"ERROR: start_idx ({start_idx}) > end_idx ({end_idx}).")
    print(f"[data] processing work list indices in [{start_idx}, {end_idx}]")

    try:
        for i in range(start_idx, end_idx + 1):
            stem, item = work_list[i]
            masks = load_masks_for_stem(args.patch_dir, stem)
            if masks is None or masks.shape[0] == 0:
                print(f"[{i}] No masks for stem {stem}; skipping.")
                continue

            print("\n" + "=" * 80)
            print(f"Processing work index {i} (stem={stem}, question_id={item.get('question_id')})")
            print("=" * 80)

            dp = seed_bench_item_to_dict(item, args.seed_bench_root)

            # First, downscale if any dimension > 1024
            orig_img = dp["image"]
            orig_w, orig_h = orig_img.size
            downscaled_img = maybe_downscale_image(orig_img, max_side=1024)
            downscaled_w, downscaled_h = downscaled_img.size
            
            if (downscaled_w, downscaled_h) != (orig_w, orig_h):
                print(f"[downscale] image {orig_w}x{orig_h} -> {downscaled_w}x{downscaled_h}")
                # Resize masks to match downscaled image
                resized_masks = [resize_mask_to_image(m, downscaled_img) for m in masks]
                masks = np.stack(resized_masks, axis=0).astype(bool)
            
            dp["image"] = downscaled_img

            if not args.no_resize:
                # Then apply square padding if requested
                img_sq, pad_info = resize_pad_to_square(dp["image"], args.image_size)
                dp["image"] = img_sq

                resized_masks = [resize_pad_mask(m, (downscaled_w, downscaled_h), args.image_size, pad_info) for m in masks]
                masks = np.stack(resized_masks, axis=0).astype(bool)
                print(f"[resize] image->{args.image_size}x{args.image_size}, masks={masks.shape}")

            try:
                results = find_patches_baseline_target(
                    llm=llm,
                    processor=processor,
                    inpaint_pipe=inpaint_pipe,
                    inpaint_device=inpaint_device,
                    data_dict=dp,
                    masks=masks,
                    args=args,
                )
            except NsfwRetryExhausted as e:
                print(f"[skip {i}] NSFW safety triggered after retries -> skipping this image. Reason: {e}")
                continue
            except torch.OutOfMemoryError as e:
                print(f"[skip {i}] CUDA out of memory during inpainting -> skipping this image. Error: {e}")
                torch.cuda.empty_cache()
                continue

            if results.get("skipped"):
                print(f"[skip {i}] {results.get('skip_reason', 'skipped')}")
                continue

            # Always save JSON for consensus later
            json_path = os.path.join(out_dir, f"patch_selection_{i}.json")
            out_json = {
                "image_idx": i,
                "question_id": item.get("question_id"),
                "path": item.get("path"),
                "stem": stem,
                "model_dir": args.vlm_model_dir,
                "model_snapshot": model_path,
                "model_tag": model_tag,

                "question": dp["question"],
                "gold_answer": dp["gold_answer"],
                "gold_letter": dp["gold_letter"],
                "answer_choices": dp["answer_choices"],
                "spurious_attribute": dp["spurious_attribute"],

                "target_letter": results.get("baseline_majority_letter"),
                "baseline_sc_target": results.get("baseline_sc_target"),
                "final_sc_target": results.get("final_sc_target"),
                "baseline_sc_gold": results.get("baseline_sc_gold"),
                "final_sc_gold": results.get("final_sc_gold"),

                "baseline_num_extracted": results.get("baseline_num_extracted"),
                "final_num_extracted": results.get("final_num_extracted"),

                "selected_mask_indices": results.get("selected_mask_indices", []),

                "search_num_patches": args.num_patches,
                "num_samples_sc": args.num_samples_sc,
                "min_sc_drop": args.min_sc_drop,
                "require_monotonic_improvement": args.require_monotonic_improvement,

                "inpaint_model_id": args.inpaint_model_id,
                "inpaint_prompt": args.inpaint_prompt,
                "inpaint_negative_prompt": args.inpaint_negative_prompt,
                "inpaint_steps": args.inpaint_steps,
                "inpaint_guidance": args.inpaint_guidance,
                "inpaint_strength": args.inpaint_strength,

                "image_size": None if args.no_resize else args.image_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "vlm_device_arg": args.vlm_device,
                "inpaint_device_arg": args.inpaint_device,

                "vllm_settings": {
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "qwen25_forced_fp8_quantization": ("qwen2.5" in model_path.lower() or "qwen25" in model_path.lower()),
                    "calculate_kv_scales_for_qwen25": ("qwen2.5" in model_path.lower() or "qwen25" in model_path.lower()),
                },
            }
            with open(json_path, "w") as f:
                json.dump(out_json, f, indent=2)
            print(f"[json] saved {json_path}")

            # Save viz
            viz_path = os.path.join(out_dir, f"viz_{i}.png")
            create_comparison_visualization(
                original_image=dp["image"],
                perturbed_image=results["final_image"],
                question=dp["question"],
                gold_answer=dp["gold_answer"],
                answer_choices=dp["answer_choices"],
                baseline_majority_letter=results.get("baseline_majority_letter"),
                final_majority_letter=results.get("final_majority_letter"),
                baseline_sc_target=results.get("baseline_sc_target"),
                baseline_sc_gold=results.get("baseline_sc_gold"),
                final_sc_target=results.get("final_sc_target"),
                final_sc_gold=results.get("final_sc_gold"),
                output_path=viz_path,
            )
        del results, dp, masks, orig_img, downscaled_img
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Exiting... attempting clean shutdown.")
    finally:
        _cleanup_distributed()

    print("\n" + "=" * 80)
    print("Completed baseline-target inpainting patch search.")
    print("=" * 80)


if __name__ == "__main__":
    main()
