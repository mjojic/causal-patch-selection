#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ---- Make vLLM/Transformers strictly offline + avoid compile stalls ----
os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# ----------------------------
# Early GPU parsing + remap
# ----------------------------
def _early_parse_devices(argv=None) -> Tuple[int, int]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vlm_device", type=int, required=True)
    p.add_argument("--inpaint_device", type=int, required=True)
    args, _ = p.parse_known_args(argv)
    return args.vlm_device, args.inpaint_device


_vlm_dev, _inp_dev = _early_parse_devices()
_shared_gpu = _vlm_dev == _inp_dev
_visible_vlm_cuda_idx = 0
_visible_inpaint_cuda_idx = 0 if _shared_gpu else 1

# If both roles use the same physical GPU, expose it once so both stacks share
# the same visible cuda:0 inside this process.
if _shared_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{_vlm_dev}"
else:
    # Map: visible GPU 0 -> VLM, visible GPU 1 -> inpaint
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{_vlm_dev},{_inp_dev}"

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from diffusers import StableDiffusionInpaintPipeline

try:
    import torch.distributed as dist
except Exception:
    dist = None


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--qwen3_results_dir", type=str, required=True,
                    help="found_patches/<qwen3_tag>/ containing patch_selection_*.json")
    ap.add_argument("--qwen25_results_dir", type=str, required=True,
                    help="found_patches/<qwen25_tag>/ containing patch_selection_*.json")
    ap.add_argument("--patch_dir", type=str, required=True,
                    help="Directory containing <stem>_masks.npz (seed bench segmentation output)")
    ap.add_argument("--output_dir", type=str, default="consensus_out")
    ap.add_argument(
        "--questions_json",
        type=str,
        default="/mnt/arc/zhaonan2/blind_project/datasets/seed_bench/seed_bench_image_questions.json",
        help="Path to seed bench questions JSON.",
    )
    ap.add_argument(
        "--seed_bench_root",
        type=str,
        default="/mnt/arc/zhaonan2/blind_project/datasets/seed_bench",
        help="Root directory for seed bench; image path = seed_bench_root + item['path'].",
    )

    # Models
    ap.add_argument("--qwen3_model_dir", type=str, required=True,
                    help="Full path to models--Qwen--Qwen3-VL-32B-Instruct-FP8 (or .../hub/models--...)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)

    # SC params
    ap.add_argument("--num_samples_sc", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024)

    # Inpainting params
    ap.add_argument("--inpaint_model_id", type=str,
                    default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--inpaint_steps", type=int, default=30)
    ap.add_argument("--inpaint_guidance", type=float, default=7.5)
    ap.add_argument("--inpaint_strength", type=float, default=1.0)

    # Adversarial diffusion prompt generation (Qwen3)
    ap.add_argument("--use_qwen_diffusion_prompt", action="store_true",
                    help="If set, use Qwen3 to generate a per-image adversarial prompt (recommended).")
    ap.add_argument("--fallback_inpaint_prompt", type=str,
                    default="Fill the masked region naturally, removing whatever objects appear there. Preserve the rest of the image exactly.")
    ap.add_argument("--fallback_negative_prompt", type=str,
                    default="text, watermark, logo, artifacts, distortion, blur")
    ap.add_argument("--qwen_prompt_temperature", type=float, default=0.2)
    ap.add_argument("--qwen_prompt_top_p", type=float, default=0.9)
    ap.add_argument("--qwen_prompt_max_tokens", type=int, default=256)

    # Devices (required early)
    ap.add_argument("--vlm_device", type=int, required=True)
    ap.add_argument("--inpaint_device", type=int, required=True)

    # Optional: limit indices for speed
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)

    return ap.parse_args()


# ----------------------------
# HF cache helpers
# ----------------------------
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
    Your layouts:
      /mnt/shared/shared_hf_home/models--...
      /mnt/shared/shared_hf_home/hub/models--...
    Hub cache root:
      /mnt/shared/shared_hf_home/hub
    """
    p = Path(models_dir)
    if p.parent.name == "hub":
        return str(p.parent)
    return str(p.parent / "hub")


def _filter_kwargs_for_llm(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(LLM.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


# ----------------------------
# Dataset (seed bench)
# ----------------------------
def seed_bench_item_to_dict(item: Dict[str, Any], seed_bench_root: str) -> Dict[str, Any]:
    """
    Map a seed bench JSON item to the same data_dict shape expected by the pipeline.
    """
    img_path = os.path.join(seed_bench_root, item["path"])
    image = Image.open(img_path).convert("RGB")

    question_raw = item["question"]
    ground_truth = str(item.get("ground_truth", "")).strip()
    gold_letter = ""
    if len(ground_truth) >= 1:
        c = ground_truth.lstrip("(").strip()[0].upper()
        if c in "ABCDE":
            gold_letter = c

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
                return str(obj["final_answer_letter"]).strip().upper()
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


def self_consistency_single_image(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    image: Image.Image,
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, Any]:
    question = make_text_prompt(data_dict["question"], data_dict["answer_choices"])

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
    }]
    vllm_input = prepare_inputs_for_vllm(messages, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples_sc,
        max_tokens=max_tokens,
    )

    out = llm.generate([vllm_input], sampling_params=sampling_params)[0]

    num_extracted = 0
    letter_counts: Dict[str, int] = {}
    for gen in out.outputs:
        parsed = parse_qwen_output(gen.text)
        if parsed is not None:
            num_extracted += 1
            letter_counts[parsed] = letter_counts.get(parsed, 0) + 1

    majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
    gold_letter = data_dict["gold_letter"]
    gold_sc = (letter_counts.get(gold_letter, 0) / num_extracted) if (num_extracted > 0 and gold_letter) else None

    return {
        "num_extracted": num_extracted,
        "letter_counts": letter_counts,
        "majority_letter": majority_letter,
        "gold_sc": gold_sc,
    }


# ----------------------------
# Qwen adversarial prompt gen
# ----------------------------
def apply_black_mask(img: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    arr = np.array(img).copy()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr[mask_bool] = 0
    return Image.fromarray(arr)


def _letter_to_choice_text(letter: Optional[str], answer_choices: List[str]) -> str:
    if letter is None:
        return "N/A"
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(answer_choices):
        return answer_choices[idx]
    return "N/A"


def qwen_diffusion_prompt_one(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    masked_image: Image.Image,
    baseline_letter: str,
    baseline_answer_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, str]:
    q = data_dict["question"]
    choices = data_dict["answer_choices"]

    instruction = (
        "You are helping generate an image-edit instruction for an inpainting model.\n\n"
        "Task:\n"
        f"Given the VQA question and answer choices, the baseline answer is {baseline_letter} ({baseline_answer_text}).\n"
        f"Question: {q}\n"
        f"Answer choices: {choices}\n"
        f"Think about what could be placed INSIDE the blacked-out region to make {baseline_letter} less likely.\n\n"
        "Constraints:\n"
        "- Only describe what should appear INSIDE the region.\n"
        "- Keep it short, ~10 words, concrete, visual.\n"
        "- Ensure positive and negative prompts are cohesive.\n"
        "- Do not put negative instructions in the positive prompt.\n"
        "- Do not put positive additions in the negative prompt.\n"
        "- Preserve everything outside the region.\n\n"
        "Return JSON ONLY:\n"
        "{\"inpaint_prompt\": \"...\", \"negative_prompt\": \"...\"}\n\n"
    )

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": masked_image}, {"type": "text", "text": instruction}],
    }]
    vllm_input = prepare_inputs_for_vllm(messages, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=1,
        max_tokens=max_tokens,
    )
    out = llm.generate([vllm_input], sampling_params=sampling_params)[0]
    raw = out.outputs[0].text if out.outputs else ""

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0).replace("'", '"'))
            ip = str(obj.get("inpaint_prompt", "")).strip()
            nprompt = str(obj.get("negative_prompt", "")).strip()
            if ip:
                return {"inpaint_prompt": ip, "negative_prompt": nprompt}
        except Exception:
            pass

    return {
        "inpaint_prompt": "A plausible object that contradicts the baseline answer.",
        "negative_prompt": "blurry, watermark, distorted, artifacts, text",
    }


# ----------------------------
# Inpainting
# ----------------------------
def _resolve_inpaint_model_path(model_id: str) -> str:
    """If model_id is a HuggingFace cache repo root (has snapshots/ and refs/), resolve to snapshots/<revision>."""
    path = Path(model_id)
    if not path.is_dir():
        return model_id
    refs_dir = path / "refs"
    snapshots_dir = path / "snapshots"
    main_ref = refs_dir / "main"
    if refs_dir.is_dir() and snapshots_dir.is_dir() and main_ref.is_file():
        revision = main_ref.read_text().strip()
        snapshot_path = snapshots_dir / revision
        if snapshot_path.is_dir():
            return str(snapshot_path)
    return model_id


def load_inpaint_pipe(model_id: str):
    device = f"cuda:{_visible_inpaint_cuda_idx}" if torch.cuda.is_available() else "cpu"
    resolved = _resolve_inpaint_model_path(model_id)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        resolved,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    return pipe, device


def mask_bool_to_pil_L(mask_bool: np.ndarray) -> Image.Image:
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def inpaint_one(
    pipe: StableDiffusionInpaintPipeline,
    device: str,
    base_image: Image.Image,
    mask_bool: np.ndarray,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    strength: float,
    seed: int,
) -> Image.Image:
    use_cuda = device.startswith("cuda")
    gen = torch.Generator(device=device).manual_seed(seed) if use_cuda else None

    w, h = base_image.size
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    with torch.inference_mode():
        res = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=mask_bool_to_pil_L(mask_bool),
            height=h8,
            width=w8,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            strength=strength,
            generator=gen,
        )
    if use_cuda:
        torch.cuda.synchronize(torch.device(device))
    return res.images[0]


# ----------------------------
# Image downscaling helpers
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


# ----------------------------
# Mask IO (seed bench: stem-based)
# ----------------------------
def load_masks_for_stem(patch_dir: str, stem: str) -> Optional[np.ndarray]:
    mask_path = os.path.join(patch_dir, f"{stem}_masks.npz")
    if not os.path.exists(mask_path):
        return None
    data = np.load(mask_path)
    return data["masks"]  # (N,H,W) bool


def save_mask_png(mask_bool: np.ndarray, out_path: str):
    im = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    im.save(out_path)


# ----------------------------
# vLLM init (Qwen3 only)
# ----------------------------
def resolve_hf_snapshot_dir(model_dir: str) -> str:
    """
    Accept either:
      - a resolved snapshot dir containing config.json
      - a HF cache dir containing snapshots/<hash>/

    Return: snapshot dir path that contains config.json
    """
    p = Path(model_dir)

    # Already a snapshot-like dir
    if (p / "config.json").exists():
        return str(p)

    snaps_root = p / "snapshots"
    if not snaps_root.exists():
        raise FileNotFoundError(f"No snapshots/ in: {model_dir}")

    snaps = sorted(
        [x for x in snaps_root.iterdir() if x.is_dir()],
        key=lambda x: x.stat().st_mtime,
    )
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under: {snaps_root}")

    # pick most recently modified snapshot that has config.json
    for s in reversed(snaps):
        if (s / "config.json").exists():
            return str(s)

    raise FileNotFoundError(f"No snapshot under {snaps_root} contained config.json")


def _prefer_hub_models_dir(models_dir: str) -> str:
    """
    If user gives /mnt/shared/shared_hf_home/models--...,
    prefer /mnt/shared/shared_hf_home/hub/models--... only when that path
    actually contains the model (snapshots/ or config.json). Otherwise use
    the path the user gave (models may live directly under shared_hf_home).
    """
    p = Path(models_dir)
    # already in .../hub/models--...
    if p.parent.name == "hub":
        return str(p)

    hub_candidate = p.parent / "hub" / p.name
    if hub_candidate.exists():
        # Only prefer hub when it actually has the model layout
        if (hub_candidate / "config.json").exists() or (hub_candidate / "snapshots").exists():
            return str(hub_candidate)
        # hub path exists but has no model; use user path
        return str(p)

    return str(p)


def init_qwen3_vllm_engine(args) -> Tuple[LLM, AutoProcessor, str, str]:
    """
    Offline-safe init:
      - loads AutoProcessor from a *local snapshot path* (never repo_id)
      - loads vLLM from a *local snapshot path* (never repo_id) so tokenizer stays local
      - keeps max_model_len from being dropped (no signature filter)
      - drops unsupported kwargs only if vLLM complains
    """
    # Prefer hub cache layout if available, then resolve snapshot
    hub_models_dir = _prefer_hub_models_dir(args.qwen3_model_dir)
    snapshot_dir = resolve_hf_snapshot_dir(hub_models_dir)

    # Hub root for any ancillary resolution (still local)
    hub_cache = _shared_hf_hub_cache(hub_models_dir)

    print(f"[qwen3] models_dir={hub_models_dir}")
    print(f"[qwen3] snapshot_dir={snapshot_dir}")
    print(f"[qwen3] hub_cache={hub_cache}")

    # Processor must be loaded from LOCAL snapshot to avoid hub API calls in offline mode
    processor = AutoProcessor.from_pretrained(
        snapshot_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    llm_kwargs: Dict[str, Any] = dict(
        # Use local snapshot for BOTH model and tokenizer to keep transformers fully local/offline
        model=snapshot_dir,
        tokenizer=snapshot_dir,

        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=True,
        dtype="auto",

        # optional; may be ignored by your vLLM build
        trust_remote_code=True,
        disable_log_stats=True,
    )

    # IMPORTANT: don't signature-filter; instead drop only if rejected.
    while True:
        try:
            print(f"[vLLM Qwen3] init kwargs: {llm_kwargs}")
            llm = LLM(**llm_kwargs)
            break
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m:
                raise
            bad = m.group(1)
            print(f"[vLLM Qwen3] WARNING: dropping unsupported kwarg: {bad}")
            llm_kwargs.pop(bad, None)

    # Return “repo_id” field just for logging; we’re intentionally local here.
    repo_id = "(local_snapshot)"
    return llm, processor, repo_id, hub_cache

def _cleanup_dist():
    if dist is None:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# ----------------------------
# JSON indexing
# ----------------------------
def index_patch_jsons(results_dir: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    paths = sorted(glob.glob(os.path.join(results_dir, "patch_selection_*.json")))
    for p in paths:
        m = re.search(r"patch_selection_(\d+)\.json$", p)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            with open(p, "r") as f:
                obj = json.load(f)
            out[idx] = obj
        except Exception:
            continue
    return out


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
        if letter is None:
            return "N/A"
        idx = ord(letter) - ord("A")
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
        f"SC wrt baseline target: {_fmt(baseline_sc_target)}",
        f"SC wrt gold: {_fmt(baseline_sc_gold)}",
    ]
    right_caption = [
        "Post (inpainted consensus mask)",
        f"Majority answer: {final_answer_text}",
        f"SC wrt baseline target: {_fmt(final_sc_target)}",
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


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if _shared_gpu and args.tensor_parallel_size != 1:
        raise SystemExit(
            "ERROR: when --vlm_device and --inpaint_device are the same GPU, "
            "--tensor_parallel_size must be 1."
        )

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    expected_visible_gpus = 1 if _shared_gpu else 2
    print(f"torch sees {torch.cuda.device_count()} GPUs (expected {expected_visible_gpus}).")
    if torch.cuda.is_available():
        print(
            f"vLLM GPU (physical cuda:{args.vlm_device} -> visible cuda:{_visible_vlm_cuda_idx}): "
            f"{torch.cuda.get_device_name(_visible_vlm_cuda_idx)}"
        )
        print(
            f"inpaint GPU (physical cuda:{args.inpaint_device} -> visible cuda:{_visible_inpaint_cuda_idx}): "
            f"{torch.cuda.get_device_name(_visible_inpaint_cuda_idx)}"
        )

    # Index JSONs
    q3 = index_patch_jsons(args.qwen3_results_dir)
    q25 = index_patch_jsons(args.qwen25_results_dir)
    common = sorted(set(q3.keys()) & set(q25.keys()))
    print(f"[json] qwen3={len(q3)} qwen2.5={len(q25)} common={len(common)}")

    # Load seed bench questions and build stem -> first item lookup
    print("[data] loading seed bench questions...")
    with open(args.questions_json, "r") as f:
        all_items = json.load(f)
    stem_to_item: Dict[str, Dict[str, Any]] = {}
    for item in all_items:
        s = Path(item["path"]).stem
        if s not in stem_to_item:
            stem_to_item[s] = item
    print(f"[data] {len(stem_to_item)} unique image stems in questions JSON")

    n_common = len(common)
    # start_idx/end_idx are image indices (from JSON filenames), not positions in common
    start_idx = args.start_idx
    end_idx = max(common) if args.end_idx < 0 else min(args.end_idx, max(common))
    in_range = [idx for idx in common if start_idx <= idx <= end_idx]
    print(f"[data] {n_common} common indices, {len(in_range)} in index range [{start_idx}, {end_idx}]")

    # Init Qwen3 engine (for prompt-gen + SC eval)
    print("[qwen3] init vLLM + processor...")
    llm, processor, repo_id, hub_cache = init_qwen3_vllm_engine(args)
    print(f"[qwen3] repo_id={repo_id} cache={hub_cache}")

    # Init inpaint
    print("[inpaint] init pipeline...")
    inpaint_pipe, inpaint_device = load_inpaint_pipe(args.inpaint_model_id)
    print(f"[inpaint] device={inpaint_device}")

    kept = 0
    skipped_no_selection = 0
    skipped_no_consensus = 0
    for idx in in_range:
        sel3 = q3[idx].get("selected_mask_indices", []) or []
        sel25 = q25[idx].get("selected_mask_indices", []) or []
        if len(sel3) == 0 or len(sel25) == 0:
            skipped_no_selection += 1
            continue

        consensus = sorted(set(sel3) & set(sel25))
        if len(consensus) == 0:
            skipped_no_consensus += 1
            continue

        # Resolve stem from search result JSONs (both should agree)
        stem = q3[idx].get("stem") or q25[idx].get("stem")
        if not stem:
            print(f"[{idx}] No stem in search result JSONs; skipping.")
            continue

        masks = load_masks_for_stem(args.patch_dir, stem)
        if masks is None or masks.shape[0] == 0:
            print(f"[{idx}] No masks for stem {stem}; skipping.")
            continue
        if masks.dtype != bool:
            masks = masks.astype(bool)

        item = stem_to_item.get(stem)
        if item is None:
            print(f"[{idx}] No seed bench question for stem {stem}; skipping.")
            continue

        dp = seed_bench_item_to_dict(item, args.seed_bench_root)
        
        # Downscale image if any dimension > 1024
        orig_img = dp["image"]
        orig_w, orig_h = orig_img.size
        img = maybe_downscale_image(orig_img, max_side=1024)
        w, h = img.size
        
        if (w, h) != (orig_w, orig_h):
            print(f"[downscale idx={idx}] image {orig_w}x{orig_h} -> {w}x{h}")
        
        dp["image"] = img
        
        N, Hm, Wm = masks.shape
        if (Hm, Wm) != (h, w):
            print(f"[resize masks idx={idx}] masks ({Hm},{Wm}) -> image ({h},{w})")
            # Resize all masks to match downscaled image
            resized_masks = [resize_mask_to_image(masks[i], img) for i in range(N)]
            masks = np.stack(resized_masks, axis=0).astype(bool)
            Hm, Wm = h, w

        # build global mask from consensus indices
        global_mask = np.zeros((h, w), dtype=bool)
        for mi in consensus:
            if 0 <= mi < N:
                global_mask |= masks[mi]

        # Save consensus mask image
        mask_path = os.path.join(args.output_dir, f"mask_{idx}.png")
        save_mask_png(global_mask, mask_path)

        # Baseline SC on original (Qwen3)
        base_res = self_consistency_single_image(
            llm, processor, dp, img,
            args.num_samples_sc, args.temperature, args.top_p, args.max_tokens
        )
        target_letter = base_res["majority_letter"]
        base_sc_target = sc_wrt_letter(base_res["letter_counts"], base_res["num_extracted"], target_letter)
        base_sc_gold = base_res["gold_sc"]

        if target_letter is None or base_sc_target is None:
            continue

        # Qwen adversarial diffusion prompt (same logic as your search)
        if args.use_qwen_diffusion_prompt:
            masked_for_qwen = apply_black_mask(img, global_mask)
            target_text = _letter_to_choice_text(target_letter, dp["answer_choices"])
            pobj = qwen_diffusion_prompt_one(
                llm=llm,
                processor=processor,
                data_dict=dp,
                masked_image=masked_for_qwen,
                baseline_letter=target_letter,
                baseline_answer_text=target_text,
                temperature=args.qwen_prompt_temperature,
                top_p=args.qwen_prompt_top_p,
                max_tokens=args.qwen_prompt_max_tokens,
            )
            inpaint_prompt = pobj.get("inpaint_prompt", "") or args.fallback_inpaint_prompt
            neg_prompt = pobj.get("negative_prompt", "") or args.fallback_negative_prompt
        else:
            pobj = None
            inpaint_prompt = args.fallback_inpaint_prompt
            neg_prompt = args.fallback_negative_prompt

        # Inpaint consensus region once
        out_img = inpaint_one(
            pipe=inpaint_pipe,
            device=inpaint_device,
            base_image=img,
            mask_bool=global_mask,
            prompt=inpaint_prompt,
            negative_prompt=neg_prompt,
            num_steps=args.inpaint_steps,
            guidance=args.inpaint_guidance,
            strength=args.inpaint_strength,
            seed=args.seed + 12345 + idx,
        )

        # SC on inpainted image (Qwen3)
        fin_res = self_consistency_single_image(
            llm, processor, dp, out_img,
            args.num_samples_sc, args.temperature, args.top_p, args.max_tokens
        )
        fin_majority = fin_res["majority_letter"]
        fin_sc_target = sc_wrt_letter(fin_res["letter_counts"], fin_res["num_extracted"], target_letter)
        fin_sc_gold = sc_wrt_letter(fin_res["letter_counts"], fin_res["num_extracted"], dp["gold_letter"])

        # Save viz
        viz_path = os.path.join(args.output_dir, f"viz_{idx}.png")
        create_comparison_visualization(
            original_image=img,
            perturbed_image=out_img,
            question=dp["question"],
            gold_answer=dp["gold_answer"],
            answer_choices=dp["answer_choices"],
            baseline_majority_letter=target_letter,
            final_majority_letter=fin_majority,
            baseline_sc_target=base_sc_target,
            baseline_sc_gold=base_sc_gold,
            final_sc_target=fin_sc_target,
            final_sc_gold=fin_sc_gold,
            output_path=viz_path,
        )

        # Save JSON
        out_json = {
            "image_idx": idx,
            "stem": stem,
            "question_id": item.get("question_id"),
            "path": item.get("path"),
            "consensus_mask_indices": consensus,
            "qwen3_selected_mask_indices": sel3,
            "qwen25_selected_mask_indices": sel25,

            "question": dp["question"],
            "gold_answer": dp["gold_answer"],
            "gold_letter": dp["gold_letter"],
            "answer_choices": dp["answer_choices"],
            "spurious_attribute": dp["spurious_attribute"],

            "baseline_majority_letter": target_letter,
            "baseline_sc_target": base_sc_target,
            "baseline_sc_gold": base_sc_gold,
            "baseline_num_extracted": base_res["num_extracted"],
            "baseline_letter_counts": base_res["letter_counts"],

            "final_majority_letter": fin_majority,
            "final_sc_target": fin_sc_target,
            "final_sc_gold": fin_sc_gold,
            "final_num_extracted": fin_res["num_extracted"],
            "final_letter_counts": fin_res["letter_counts"],

            "use_qwen_diffusion_prompt": bool(args.use_qwen_diffusion_prompt),
            "qwen_prompt_object": pobj,
            "inpaint_prompt_used": inpaint_prompt,
            "negative_prompt_used": neg_prompt,
            "inpaint_model_id": args.inpaint_model_id,
            "inpaint_steps": args.inpaint_steps,
            "inpaint_guidance": args.inpaint_guidance,
            "inpaint_strength": args.inpaint_strength,

            "paths": {
                "mask_png": mask_path,
                "viz_png": viz_path,
                "qwen3_results_json": os.path.join(args.qwen3_results_dir, f"patch_selection_{idx}.json"),
                "qwen25_results_json": os.path.join(args.qwen25_results_dir, f"patch_selection_{idx}.json"),
            },

            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vlm_device_arg": args.vlm_device,
            "inpaint_device_arg": args.inpaint_device,
            "qwen3_repo_id": repo_id,
            "hf_cache_dir": hub_cache,
            "sc_params": {
                "num_samples_sc": args.num_samples_sc,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
        }

        json_path = os.path.join(args.output_dir, f"consensus_{idx}.json")
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=2)

        kept += 1
        print(f"[ok {idx}] consensus={consensus} saved: {viz_path}, {mask_path}, {json_path}")

    print(f"\nDone. Wrote outputs for {kept} images with non-empty consensus.")
    print(f"  (Skipped: {skipped_no_selection} with missing selection in one model, {skipped_no_consensus} with no overlap between models.)")
    _cleanup_dist()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Exiting.")
        _cleanup_dist()