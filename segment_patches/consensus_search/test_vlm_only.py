#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test script for VLM inference only (no diffusion/inpainting).
Use this to test Qwen 2.5 VL 72B memory consumption and FP4/FP8 quantization.
"""

import os
import sys
import argparse

# ----------------------------
# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE any CUDA-related imports
# ----------------------------
def _early_parse_device(argv=None):
    """Parse GPU device argument before any imports that might initialize CUDA."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", type=int, default=0)
    args, _ = p.parse_known_args(argv)
    return args.device

_device = _early_parse_device()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_device)
print(f"[GPU config] Using GPU {_device} (visible as cuda:0)")

os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Now safe to import everything else
import re
import json
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import datasets
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams


# ----------------------------
# Helper functions
# ----------------------------
def _guess_repo_id_from_models_dir(models_dir: str) -> str:
    """/.../models--ORG--REPO  ->  ORG/REPO"""
    base = Path(models_dir).name
    m = re.match(r"models--([^/]+)--(.+)$", base)
    if not m:
        raise ValueError(f"Can't infer repo id from: {models_dir}")
    return f"{m.group(1)}/{m.group(2)}"


def _shared_hf_hub_cache(models_dir: str) -> str:
    """Return the HF hub cache directory."""
    p = Path(models_dir)
    if p.parent.name == "hub":
        return str(p.parent)
    return str(p.parent / "hub")


def resolve_hf_snapshot_dir(model_dir: str) -> str:
    """Resolve to the directory containing config.json."""
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
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
    return img.resize((new_w, new_h), resample=resample)


# ----------------------------
# VLM initialization with FP4/FP8 quantization
# ----------------------------
def init_vllm_engine(args) -> Tuple[LLM, str]:
    """
    Initialize vLLM engine with FP4 quantization (fallback to FP8).
    
    Memory consumption sources:
      1. Model weights: quantization (fp4 < fp8 < bf16/fp16)
      2. KV cache: kv_cache_dtype
      3. Batch sizes: num_samples (affects peak memory)
      4. Sequence length: max_model_len (affects KV cache size)
      5. Image size: affects vision token count
    """
    model_path = resolve_hf_snapshot_dir(args.model_dir)
    lower = model_path.lower()
    is_qwen25 = ("qwen2.5" in lower) or ("qwen25" in lower)

    repo_id = _guess_repo_id_from_models_dir(args.model_dir)
    hub_cache = _shared_hf_hub_cache(args.model_dir)

    print(f"[vLLM] Model path: {model_path}")
    print(f"[vLLM] Repo ID: {repo_id}")
    print(f"[vLLM] Hub cache: {hub_cache}")
    print(f"[vLLM] Is Qwen2.5: {is_qwen25}")

    llm_kwargs: Dict[str, Any] = dict(
        model=repo_id,
        tokenizer=repo_id,
        download_dir=hub_cache,
        hf_cache_dir=hub_cache,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=False,
        dtype="auto",
        trust_remote_code=True,
        disable_log_stats=True,
    )

    if is_qwen25:
        # Memory optimization for Qwen2.5-VL-72B:
        # - FP8 quantization for weights (vLLM 0.11.2 doesn't support direct FP4)
        # - FP8 KV cache to reduce memory further
        # - calculate_kv_scales=True required for FP8 KV cache
        print(f"[vLLM] Using FP8 quantization for Qwen2.5-VL-72B")
        llm_kwargs["quantization"] = "fp8"
        llm_kwargs["kv_cache_dtype"] = "fp8"
        llm_kwargs["calculate_kv_scales"] = True

    # Try to create engine, dropping unsupported kwargs if needed
    actual_quantization = None
    
    while True:
        try:
            print(f"[vLLM] init kwargs: {llm_kwargs}")
            llm = LLM(**llm_kwargs)
            actual_quantization = llm_kwargs.get("quantization", "none")
            return llm, actual_quantization
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
# Inference helpers
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
        json_str = m.group(0).replace("'", '"')
        json_str = re.sub(r",\s*}", "}", json_str)
        try:
            obj = json.loads(json_str)
            if "final_answer_letter" in obj:
                value = str(obj["final_answer_letter"]).strip().upper()
                if len(value) == 1 and value in "ABCDE":
                    return value
        except Exception:
            pass

    m2 = re.search(r"final_answer_letter\s*[:=]\s*['\"]?([A-Ea-e])['\"]?", text)
    if m2:
        return m2.group(1).upper()

    m3 = re.search(r"^\s*([A-E])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()

    return None


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


def run_inference(
    llm: LLM,
    processor: AutoProcessor,
    image: Image.Image,
    question: str,
    answer_choices: List[str],
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run single inference with self-consistency sampling."""
    prompt = make_text_prompt(question, answer_choices)
    
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    vllm_input = prepare_inputs_for_vllm(messages, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples,
        max_tokens=max_tokens,
    )

    start_time = time.time()
    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    inference_time = time.time() - start_time

    # Parse outputs
    letter_counts: Dict[str, int] = {}
    num_extracted = 0
    raw_outputs = []
    
    for gen in outputs[0].outputs:
        raw_outputs.append(gen.text[:200] + "..." if len(gen.text) > 200 else gen.text)
        parsed = parse_qwen_output(gen.text)
        if parsed is not None:
            num_extracted += 1
            letter_counts[parsed] = letter_counts.get(parsed, 0) + 1

    majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None

    return {
        "num_extracted": num_extracted,
        "letter_counts": letter_counts,
        "majority_letter": majority_letter,
        "inference_time": inference_time,
        "raw_outputs": raw_outputs[:3],  # First 3 for debugging
    }


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Test VLM inference only (no diffusion)")
    
    # GPU
    ap.add_argument("--device", type=int, default=0, help="GPU device ID")
    
    # Model
    ap.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct",
        help="Path to model directory",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--max_model_len", type=int, default=11000)
    ap.add_argument("--seed", type=int, default=0)
    
    # Inference
    ap.add_argument("--num_samples", type=int, default=5, help="Number of samples for self-consistency")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024)
    
    # Data
    ap.add_argument("--num_examples", type=int, default=3, help="Number of examples to test")
    ap.add_argument("--start_idx", type=int, default=0, help="Starting index in dataset")
    ap.add_argument("--max_image_size", type=int, default=1024, help="Max image dimension (downscale if larger)")
    
    return ap.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("VLM-ONLY TEST (No Diffusion)")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_dir}")
    print(f"tensor_parallel_size: {args.tensor_parallel_size}")
    print(f"gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"max_model_len: {args.max_model_len}")
    print(f"num_samples: {args.num_samples}")
    print(f"max_image_size: {args.max_image_size}")
    print("=" * 80)
    
    # Check GPU memory before loading
    if torch.cuda.is_available():
        print(f"\n[GPU] Name: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Total memory: {total_mem:.2f} GB")
    
    # Load processor
    model_path = resolve_hf_snapshot_dir(args.model_dir)
    print(f"\n[Processor] Loading from: {model_path}")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    except OSError:
        repo_id = _guess_repo_id_from_models_dir(args.model_dir)
        hub_cache = _shared_hf_hub_cache(args.model_dir)
        print(f"[Processor] Falling back to repo_id={repo_id}")
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=True, cache_dir=hub_cache
        )
    
    # Initialize VLM engine
    print("\n[vLLM] Initializing engine...")
    init_start = time.time()
    llm, actual_quantization = init_vllm_engine(args)
    init_time = time.time() - init_start
    print(f"[vLLM] Engine initialized in {init_time:.2f}s with quantization: {actual_quantization}")
    
    # Check GPU memory after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[GPU] After model load - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # Load test data
    print("\n[Data] Loading NaturalBench dataset...")
    ds = datasets.load_dataset("BaiqiL/NaturalBench-lmms-eval", split="test")
    print(f"[Data] Dataset size: {len(ds)}")
    
    # Run inference on a few examples
    print(f"\n[Test] Running inference on {args.num_examples} examples...")
    
    for i in range(args.num_examples):
        idx = args.start_idx + i
        if idx >= len(ds):
            print(f"Reached end of dataset at idx={idx}")
            break
            
        print(f"\n{'='*60}")
        print(f"Example {i+1}/{args.num_examples} (dataset idx={idx})")
        print(f"{'='*60}")
        
        dp = ds[idx]
        image = dp["Image"]
        question = dp["Question"]
        q_type = dp["Question_Type"]
        answer = dp["Answer"]
        
        # Downscale image if needed
        orig_size = image.size
        image = maybe_downscale_image(image, args.max_image_size)
        new_size = image.size
        if orig_size != new_size:
            print(f"[Image] Downscaled: {orig_size} -> {new_size}")
        else:
            print(f"[Image] Size: {orig_size}")
        
        # Prepare answer choices
        if q_type == "yes_no":
            answer_choices = ["Yes", "No"]
        else:
            answer_choices = ["Option A", "Option B"]
            if "Option:" in question:
                head, tail = question.split("Option:", 1)
                question = head.strip()
                segs = [s.strip() for s in tail.split(";") if s.strip()]
                for s in segs:
                    if s.startswith("A:"):
                        answer_choices[0] = s[2:].strip()
                    elif s.startswith("B:"):
                        answer_choices[1] = s[2:].strip()
        
        print(f"[Q] {question[:100]}...")
        print(f"[Choices] {answer_choices}")
        print(f"[Gold] {answer}")
        
        # Run inference
        try:
            result = run_inference(
                llm=llm,
                processor=processor,
                image=image,
                question=question,
                answer_choices=answer_choices,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            
            print(f"\n[Result] Inference time: {result['inference_time']:.2f}s")
            print(f"[Result] Extracted: {result['num_extracted']}/{args.num_samples}")
            print(f"[Result] Letter counts: {result['letter_counts']}")
            print(f"[Result] Majority: {result['majority_letter']}")
            
            # Check GPU memory after inference
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"[GPU] After inference - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[ERROR] CUDA Out of Memory: {e}")
            print("[TIP] Try reducing: max_model_len, num_samples, or max_image_size")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            raise
    
    print("\n" + "=" * 80)
    print("VLM-ONLY TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
