#!/usr/bin/env python3
"""
Evaluate models on the POPE benchmark using HuggingFace transformers (no vLLM).

Compares up to 3 model configurations:
1. Base Qwen 3 VL 8B
2. Fine-tuned with VQA prediction loss + attention alignment loss
3. Fine-tuned with VQA prediction loss only

Runs greedy decoding sample-by-sample through the dataset.
Reports overall accuracy and per-category (adversarial / popular / random) accuracy.
"""

import os
import sys

# ----------------------------
# CRITICAL: Parse --gpu argument and set CUDA_VISIBLE_DEVICES BEFORE any CUDA imports
# ----------------------------
def _parse_gpu_early() -> int:
    """Parse --gpu from sys.argv before argparse to set CUDA_VISIBLE_DEVICES early."""
    gpu = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            try:
                gpu = int(sys.argv[i + 1])
            except ValueError:
                pass
            break
        elif arg.startswith("--gpu="):
            try:
                gpu = int(arg.split("=", 1)[1])
            except ValueError:
                pass
            break
    return gpu

_GPU_ID = _parse_gpu_early()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU_ID)

import argparse
import re
import json
import glob
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

import gc

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from pope_dataset_loader import POPEDataset


# -------------------------
# Environment
# -------------------------
os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# -------------------------
# Model path resolution
# -------------------------
def _guess_repo_id_from_models_dir(models_dir: str) -> str:
    """/.../models--ORG--REPO  ->  ORG/REPO"""
    base = Path(models_dir).name
    m = re.match(r"models--([^/]+)--(.+)$", base)
    if not m:
        raise ValueError(f"Can't infer repo id from: {models_dir}")
    return f"{m.group(1)}/{m.group(2)}"


def _shared_hf_hub_cache(models_dir: str) -> str:
    p = Path(models_dir)
    if p.parent.name == "hub":
        return str(p.parent)
    return str(p.parent / "hub")


def resolve_hf_snapshot_dir(model_dir: str) -> str:
    """Resolve to snapshot directory containing config.json"""
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


# -------------------------
# Prompting and parsing (POPE-specific)
# -------------------------
def make_text_prompt(question: str, answer_choices: List[str]) -> str:
    """Format a POPE yes/no question as a multiple-choice prompt."""
    prompt = (
        "Given the following question and answer choices about an image, select the best matching answer.\n\n"
        f"Question: {question}\n"
        "Answer Choices:\n"
    )
    options = ["A", "B"]
    for i, choice in enumerate(answer_choices):
        prompt += f"{options[i]}: {choice}\n"
    prompt += "\nProvide your final answer in this JSON format: {'final_answer_letter': <letter>}\n"
    return prompt


def parse_model_output(text: str) -> Optional[str]:
    """Extract answer letter (A or B) from model output."""
    m = re.search(r"\{.*?final_answer_letter.*?\}", text, re.DOTALL)
    if m:
        json_str = m.group(0).replace("'", '"')
        json_str = re.sub(r",\s*}", "}", json_str)
        try:
            obj = json.loads(json_str)
            val = str(obj.get("final_answer_letter", "")).strip().upper()
            if val in ("A", "B"):
                return val
        except Exception:
            pass

    m2 = re.search(r"final_answer_letter\s*[:=]\s*['\"]?([ABab])['\"]?", text)
    if m2:
        return m2.group(1).upper()

    m3 = re.search(r"^\s*([AB])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()

    m4 = re.search(r"(?:answer|choice)(?:\s+is)?[:\s]+([AB])\b", text, re.IGNORECASE)
    if m4:
        return m4.group(1).upper()

    lowered = text.strip().lower()
    if re.search(r"\byes\b", lowered) and not re.search(r"\bno\b", lowered):
        return "A"
    if re.search(r"\bno\b", lowered) and not re.search(r"\byes\b", lowered):
        return "B"

    return None


# -------------------------
# HuggingFace model loading / inference
# -------------------------
def load_model(
    model_path: str,
    processor: AutoProcessor,
    lora_adapter_path: Optional[str] = None,
) -> AutoModelForImageTextToText:
    """Load Qwen3-VL model, optionally with a merged LoRA adapter."""
    print(f"\n[model] Loading {model_path} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    if lora_adapter_path is not None:
        print(f"[model] Applying LoRA adapter: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
        print("[model] LoRA merged and unloaded")

    model.eval()
    print("[model] Ready")
    return model


def cleanup_model(model) -> None:
    """Delete model and free GPU memory."""
    print("\n[cleanup] Freeing GPU memory...")
    del model
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[cleanup] GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def run_inference(
    model,
    processor: AutoProcessor,
    image,
    prompt_text: str,
    max_new_tokens: int = 512,
) -> str:
    """Run a single forward pass through the model and return decoded text."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# -------------------------
# Evaluation functions
# -------------------------
def evaluate_sample(
    model,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    """Evaluate a single POPE sample with greedy decoding."""
    prompt = make_text_prompt(sample["question"], sample["answer_choices"])
    raw_output = run_inference(model, processor, sample["image"], prompt, max_new_tokens=max_tokens)

    predicted_letter = parse_model_output(raw_output)
    is_correct = (predicted_letter == sample["gold_letter"]) if predicted_letter else False

    return {
        "index": sample["index"],
        "question": sample["question"],
        "answer_choices": sample["answer_choices"],
        "predicted_letter": predicted_letter,
        "gold_letter": sample["gold_letter"],
        "gold_answer": sample["gold_answer"],
        "category": sample["category"],
        "image_source": sample["image_source"],
        "is_correct": is_correct,
        "raw_output": raw_output,
    }


def evaluate_model(
    model,
    processor: AutoProcessor,
    dataset: POPEDataset,
    model_name: str,
    args,
) -> Dict[str, Any]:
    """Evaluate a model on the POPE dataset with per-category tracking."""
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 80}")

    results: List[Dict[str, Any]] = []
    correct = 0
    category_stats: Dict[str, Dict[str, int]] = {}

    start_idx = args.start_idx
    end_idx = min(args.end_idx, len(dataset))
    eval_indices = list(range(start_idx, end_idx))
    num_eval = len(eval_indices)

    print(f"Evaluating {num_eval} samples: indices [{start_idx}, {end_idx})")

    iterator = tqdm(eval_indices, desc=model_name) if tqdm else eval_indices

    for idx in iterator:
        sample = dataset[idx]

        result = evaluate_sample(
            model=model,
            processor=processor,
            sample=sample,
            max_tokens=args.max_tokens,
        )

        results.append(result)

        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1

        if result["is_correct"]:
            correct += 1
            category_stats[cat]["correct"] += 1

        if not tqdm and len(results) % 50 == 0:
            acc = correct / len(results) * 100
            print(f"  [{len(results)}/{num_eval}] Current accuracy: {acc:.2f}%")

    accuracy = correct / num_eval * 100 if num_eval else 0
    print(f"\n[{model_name}] Final accuracy: {accuracy:.2f}% ({correct}/{num_eval})")
    for cat, stats in sorted(category_stats.items()):
        cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {cat}: {cat_acc:.2f}% ({stats['correct']}/{stats['total']})")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": num_eval,
        "by_category": category_stats,
        "results": results,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate models on POPE benchmark (HF, no vLLM)")

    # Model paths
    parser.add_argument("--base_model_dir", type=str,
                        default="/mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct",
                        help="Path to base model directory")
    parser.add_argument("--lora_vqa_attn_dir", type=str, default=None,
                        help="Path to LoRA adapter trained with VQA + attention alignment loss")
    parser.add_argument("--lora_vqa_only_dir", type=str, default=None,
                        help="Path to LoRA adapter trained with VQA loss only")

    # Dataset
    parser.add_argument("--pope_cache_dir", type=str,
                        default="/mnt/shared/shared_hf_home/datasets",
                        help="Cache directory for HuggingFace POPE dataset")
    parser.add_argument("--max_image_side", type=int, default=1024,
                        help="Max image dimension; larger images are downscaled")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in dataset (inclusive)")
    parser.add_argument("--end_idx", type=int, default=9000,
                        help="End index in dataset (exclusive)")

    # Inference settings
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum new tokens to generate per sample")
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--gpu", type=int, default=0)

    # Output
    parser.add_argument("--output_json", type=str, default="pope_results_hf.json",
                        help="Output JSON file for results")

    # Model selection
    parser.add_argument("--eval_base", action="store_true", default=True,
                        help="Evaluate base model")
    parser.add_argument("--no_eval_base", action="store_false", dest="eval_base",
                        help="Skip base model evaluation")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    base_snapshot = resolve_hf_snapshot_dir(args.base_model_dir)

    print("=" * 80)
    print("POPE BENCHMARK EVALUATION  (HuggingFace, greedy)")
    print("=" * 80)
    print(f"Base model: {base_snapshot}")
    print(f"LoRA (VQA+Attn): {args.lora_vqa_attn_dir or 'None'}")
    print(f"LoRA (VQA only): {args.lora_vqa_only_dir or 'None'}")
    print(f"POPE cache dir: {args.pope_cache_dir}")
    print(f"Eval range: [{args.start_idx}, {args.end_idx})")
    print("=" * 80)

    # Load processor
    print("\n[processor] Loading...")
    try:
        processor = AutoProcessor.from_pretrained(base_snapshot, trust_remote_code=True, local_files_only=True)
    except OSError:
        repo_id = _guess_repo_id_from_models_dir(args.base_model_dir)
        hub_cache = _shared_hf_hub_cache(args.base_model_dir)
        print(f"[processor] Fallback to repo_id={repo_id}")
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=True, cache_dir=hub_cache,
        )

    # Load POPE dataset
    print("\n[dataset] Loading POPE...")
    dataset = POPEDataset(
        cache_dir=args.pope_cache_dir,
        max_image_side=args.max_image_side,
    )

    if args.end_idx > len(dataset):
        print(f"[warning] end_idx={args.end_idx} > dataset size={len(dataset)}, adjusting...")
        args.end_idx = len(dataset)

    # ---- results container ----
    all_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": base_snapshot,
            "lora_vqa_attn": args.lora_vqa_attn_dir,
            "lora_vqa_only": args.lora_vqa_only_dir,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "max_tokens": args.max_tokens,
            "inference": "hf_greedy",
        },
        "models": {},
    }

    # ---- LoRA models ----
    if args.lora_vqa_attn_dir is not None:
        model = load_model(base_snapshot, processor, lora_adapter_path=args.lora_vqa_attn_dir)
        result = evaluate_model(model, processor, dataset, "VQA + Attention Alignment", args)
        all_results["models"]["vqa_attn"] = result
        cleanup_model(model)

    if args.lora_vqa_only_dir is not None:
        model = load_model(base_snapshot, processor, lora_adapter_path=args.lora_vqa_only_dir)
        result = evaluate_model(model, processor, dataset, "VQA Only", args)
        all_results["models"]["vqa_only"] = result
        cleanup_model(model)

    # ---- Base model ----
    if args.eval_base:
        model = load_model(base_snapshot, processor)
        result = evaluate_model(model, processor, dataset, "Base Qwen3-VL-8B", args)
        all_results["models"]["base"] = result
        cleanup_model(model)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_key, model_result in all_results["models"].items():
        name = model_result["model_name"]
        acc = model_result["accuracy"]
        cor = model_result["correct"]
        tot = model_result["total"]
        print(f"  {name:40s}: {acc:5.2f}% ({cor}/{tot})")
        for cat, stats in sorted(model_result["by_category"].items()):
            cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
            print(f"    {cat:20s}: {cat_acc:5.2f}% ({stats['correct']}/{stats['total']})")

    if "base" in all_results["models"]:
        base_acc = all_results["models"]["base"]["accuracy"]
        for model_key in ["vqa_attn", "vqa_only"]:
            if model_key in all_results["models"]:
                other_acc = all_results["models"][model_key]["accuracy"]
                diff = other_acc - base_acc
                print(f"  {all_results['models'][model_key]['model_name']} vs Base: {diff:+.2f}%")

    print("=" * 80)

    # ---- Save ----
    for model_key in all_results["models"]:
        for r in all_results["models"][model_key]["results"]:
            r.pop("image", None)

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
