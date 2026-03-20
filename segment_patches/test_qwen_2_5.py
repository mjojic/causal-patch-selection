#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset


def set_hf_home(hf_home: str):
    hf_home = os.path.abspath(hf_home)
    os.environ["HF_HOME"] = hf_home
    # Keep hub + datasets in the shared HF_HOME
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))


def resolve_hf_snapshot_dir(model_dir: str) -> str:
    """
    Accepts either:
      - a normal HF model dir containing config.json
      - a HF hub 'models--ORG--NAME' directory containing snapshots/<hash>/
    Returns a directory that actually contains config.json.
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


def pick_naturalbench_example(
    split: str,
    idx: int | None,
    image_idx: int,
    question_idx: int,
    offline: bool,
):
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # NaturalBench currently provides a 'train' split with ~1900 rows.  [oai_citation:1‡Hugging Face](https://huggingface.co/datasets/BaiqiL/NaturalBench)
    ds = load_dataset("BaiqiL/NaturalBench", split=split)

    if idx is None:
        idx = int(torch.randint(low=0, high=len(ds), size=(1,)).item())

    ex = ds[idx]

    img = ex[f"Image_{image_idx}"]
    if not isinstance(img, Image.Image):
        # HF datasets usually returns PIL images for Image features; this is just a safety net.
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")

    question = ex[f"Question_{question_idx}"]
    answer_key = f"Image_{image_idx}_Question_{question_idx}"
    gt_answer = ex.get(answer_key, None)

    return idx, img, question, gt_answer, ex


def run_transformers(model_path: str, img: Image.Image, question: str, max_new_tokens: int):
    from transformers import AutoProcessor, AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    ).eval()

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return decoded

def run_vllm(model_path, img, question, max_new_tokens, tp, fp8,
             max_model_len, gpu_memory_utilization, max_num_seqs, max_num_batched_tokens, kv_cache_dtype):

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": question}],
    }]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_cache_dtype=kv_cache_dtype,          # KV cache quantization  [oai_citation:4‡vLLM](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/?utm_source=chatgpt.com)
        calculate_kv_scales=(kv_cache_dtype.startswith("fp8")),  # helps FP8 KV cache  [oai_citation:5‡vLLM](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/?utm_source=chatgpt.com)
    )
    if fp8:
        llm_kwargs["quantization"] = "fp8"      # weight quantization (separate from KV)

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outputs = llm.generate(
        [{"prompt": prompt_text, "multi_modal_data": {"image": img}}],
        sampling,
    )
    return outputs[0].outputs[0].text

def main():
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument(
        "--model_dir",
        type=str,
        default="models--Qwen--Qwen2.5-VL-72B-Instruct",
        help="Local HF hub model folder OR resolved snapshot folder.",
    )

    # HF caching / offline
    ap.add_argument(
        "--hf_home",
        type=str,
        default=None,
        help="Path to shared HF_HOME (e.g., /mnt/shared/HF_HOME). Sets HF_HOME/HF_HUB_CACHE/HF_DATASETS_CACHE.",
    )
    ap.add_argument("--offline", action="store_true", help="Force offline mode (requires dataset/model already cached).")

    # NaturalBench selection
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--idx", type=int, default=None, help="Example index. If omitted, picks a random one.")
    ap.add_argument("--image_idx", type=int, choices=[0, 1], default=0)
    ap.add_argument("--question_idx", type=int, choices=[0, 1], default=0)

    # Generation
    ap.add_argument("--max_new_tokens", type=int, default=128)

    # Backend
    ap.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    ap.add_argument("--fp8", action="store_true", help="Try FP8 in vLLM if supported.")
    ap.add_argument("--max_model_len", type=int, default=10000)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--max_num_seqs", type=int, default=1)
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--kv_cache_dtype", type=str, default="fp8", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])

    args = ap.parse_args()

    if args.hf_home:
        set_hf_home(args.hf_home)

    model_path = resolve_hf_snapshot_dir(args.model_dir)
    print(f"[info] Model resolved to: {model_path}")

    idx, img, question, gt, ex = pick_naturalbench_example(
        split=args.split,
        idx=args.idx,
        image_idx=args.image_idx,
        question_idx=args.question_idx,
        offline=args.offline,
    )

    print(f"\n[info] NaturalBench example idx={idx} | image={args.image_idx} | question={args.question_idx}")
    print(f"[question]\n{question}")
    if gt is not None:
        print(f"[ground truth] {gt}")
    if "Question Type" in ex:
        print(f"[question type] {ex['Question Type']}")

    # Run model
    if args.backend in ("auto", "vllm"):
        try:
            out = run_vllm(
                model_path, img, question,
                args.max_new_tokens, args.tp, args.fp8,
                args.max_model_len, args.gpu_memory_utilization,
                args.max_num_seqs, args.max_num_batched_tokens,
                args.kv_cache_dtype,
            )
            print("\n=== vLLM OUTPUT ===")
            print(out)
            return
        except Exception as e:
            if args.backend == "vllm":
                raise
            print(f"\n[warn] vLLM failed ({type(e).__name__}: {e}). Falling back to Transformers...\n")

    out = run_transformers(model_path, img, question, args.max_new_tokens)
    print("\n=== TRANSFORMERS OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()