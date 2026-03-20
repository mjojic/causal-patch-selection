#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to load the Qwen 2.5 VL 72B model using vLLM.
This isolates the model loading part for testing purposes.
"""

import os
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, Any
from vllm import LLM


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


def resolve_hf_snapshot_dir(model_dir: str) -> str:
    """
    Accepts either:
      - a resolved snapshot directory containing config.json
      - a HF hub models--ORG--NAME directory containing snapshots/<hash>/
    Returns a directory that contains config.json.
    """
    p = Path(model_dir)
    if (p / "config.json").exists():
        return str(p)

    snaps = sorted(glob.glob(str(p / "snapshots" / "*")), key=lambda x: os.path.getmtime(x))
    if not snaps:
        raise FileNotFoundError(
            f"Could not find snapshots under: {model_dir}\n"
            f"Expected: {model_dir}/snapshots/<hash>/config.json"
        )
    snap = Path(snaps[-1])
    if not (snap / "config.json").exists():
        raise FileNotFoundError(f"Snapshot missing config.json: {snap}")
    return str(snap)


def init_vllm_engine(model_path: str, args) -> LLM:
    """
    Robust vLLM init that:
      - Uses local HF cache layout you have (models--ORG--REPO + shared hub cache)
      - Forces Qwen2.5-VL-72B to use fp8 weight quant + fp8 KV + calculate_kv_scales=True
      - Ensures max_model_len is actually passed (no signature-based filtering)
      - Drops only truly-unsupported kwargs if this vLLM build rejects them
    """
    lower = model_path.lower()
    is_qwen25 = ("qwen2.5" in lower) or ("qwen25" in lower)

    # Point vLLM to the repo_id so it can find tokenizer/processor assets via HF cache.
    repo_id = _guess_repo_id_from_models_dir(args.vlm_model_dir)
    hub_cache = _shared_hf_hub_cache(args.vlm_model_dir)

    llm_kwargs: Dict[str, Any] = dict(
        model=repo_id,
        tokenizer=repo_id,
        image_input_type="pixel_values",
        video_input_type="pixel_values",
        download_dir=hub_cache,
        hf_cache_dir=hub_cache,

        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=False,
        dtype="auto",
        kv_cache_dtype="auto",
        trust_remote_code=True,
        disable_log_stats=True,
    )

    if is_qwen25:
        llm_kwargs.update({
            "quantization": "fp8",
            "kv_cache_dtype": "fp8",
            "calculate_kv_scales": True,
        })

    # DO NOT signature-filter. Some vLLM versions hide real engine args from LLM.__init__ signature.
    # Instead, try and only strip unsupported kwargs if vLLM throws "unexpected keyword argument".
    while True:
        try:
            llm = LLM(**llm_kwargs)
            break
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg:
                # Extract the bad arg name
                import re
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if m:
                    bad_arg = m.group(1)
                    print(f"[vLLM] Removing unsupported kwarg: {bad_arg}")
                    llm_kwargs.pop(bad_arg, None)
                    continue
            # Re-raise if not a kwarg issue
            raise

    return llm


def parse_args():
    ap = argparse.ArgumentParser()

    # GPUs (required early)
    ap.add_argument("--vlm_device", type=int, required=True)

    # VLM / vLLM
    ap.add_argument(
        "--vlm_model_dir",
        type=str,
        required=True,
        help="Path to local HF hub folder (models--...); script will auto-resolve snapshots/<hash>/.",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


def main():
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.vlm_device)

    # Set other env vars
    os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    model_path = resolve_hf_snapshot_dir(args.vlm_model_dir)

    print(f"[model] resolved snapshot: {model_path}")
    print(f"[vLLM] initializing engine on GPU {args.vlm_device}...")

    llm = init_vllm_engine(model_path, args)

    print("[vLLM] Model loaded successfully!")
    print(f"Model type: {type(llm)}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Max model length: {args.max_model_len}")

    # Optional: Test a simple generation
    print("[test] Running a simple text generation test...")
    test_prompt = "Hello, how are you?"
    outputs = llm.generate([test_prompt], sampling_params=None)
    print(f"Test output: {outputs[0].outputs[0].text}")

    print("Test completed successfully!")


if __name__ == "__main__":
    main()