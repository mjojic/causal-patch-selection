#!/usr/bin/env python3
"""
Evaluate models on the POPE (Polling-based Object Probing Evaluation) benchmark.

Compares up to 3 model configurations:
1. Base Qwen 3 VL 8B
2. Fine-tuned with VQA prediction loss + attention alignment loss
3. Fine-tuned with VQA prediction loss only

Uses vLLM for efficient inference with optional self-consistency.
Reports overall accuracy and per-category (adversarial / popular / random) accuracy.
"""

import os
import sys

# ----------------------------
# CRITICAL: Parse --gpu argument and set CUDA_VISIBLE_DEVICES BEFORE any CUDA imports
# ----------------------------
def _parse_gpu_early() -> int:
    """Parse --gpu from sys.argv before argparse to set CUDA_VISIBLE_DEVICES early."""
    gpu = 0  # default
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

os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import re
import json
import glob
import shutil
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

import gc

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
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
    """Get hub cache directory from models directory path"""
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
        "Given the following question and answer choices about an image, do the following:\n"
        "1) Look carefully at the image and identify visual details relevant to the question.\n"
        "2) Reason about each answer choice based on what you observe.\n"
        "3) Select the single best answer choice.\n\n"
        f"Question: {question}\n"
        "Answer Choices:\n"
    )
    options = ["A", "B"]
    for i, choice in enumerate(answer_choices):
        prompt += f"{options[i]}: {choice}\n"
    prompt += "\nProvide your final answer in this JSON format: {'final_answer_letter': <letter>}\n"
    return prompt


def parse_model_output(text: str) -> Optional[str]:
    """Extract answer letter (A or B) from model output.

    Tries structured JSON first, then various heuristic patterns.  Also maps
    bare "yes"/"no" responses to A/B as a last-resort fallback.
    """
    # Try JSON format
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

    # Key-value format
    m2 = re.search(r"final_answer_letter\s*[:=]\s*['\"]?([ABab])['\"]?", text)
    if m2:
        return m2.group(1).upper()

    # Standalone letter on its own line
    m3 = re.search(r"^\s*([AB])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()

    # "Answer: X" / "The answer is X" patterns
    m4 = re.search(r"(?:answer|choice)(?:\s+is)?[:\s]+([AB])\b", text, re.IGNORECASE)
    if m4:
        return m4.group(1).upper()

    # Bare "yes" / "no" fallback
    lowered = text.strip().lower()
    if re.search(r"\byes\b", lowered):
        if not re.search(r"\bno\b", lowered):
            return "A"
    if re.search(r"\bno\b", lowered):
        if not re.search(r"\byes\b", lowered):
            return "B"

    return None


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    """Prepare inputs for vLLM inference."""
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


# -------------------------
# Evaluation functions
# -------------------------
def _parse_batch_outputs(
    batch_samples: List[Dict[str, Any]],
    batch_outputs,
    num_samples_sc: int,
) -> List[Dict[str, Any]]:
    """Parse vLLM outputs for a batch of samples into result dicts."""
    batch_results = []
    for sample, request_output in zip(batch_samples, batch_outputs):
        num_extracted = 0
        letter_counts: Dict[str, int] = {}
        raw_outputs = []

        for gen in request_output.outputs:
            raw_outputs.append(gen.text)
            parsed = parse_model_output(gen.text)
            if parsed:
                num_extracted += 1
                letter_counts[parsed] = letter_counts.get(parsed, 0) + 1

        majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
        is_correct = (majority_letter == sample["gold_letter"]) if majority_letter else False

        batch_results.append({
            "index": sample["index"],
            "question": sample["question"],
            "answer_choices": sample["answer_choices"],
            "predicted_letter": majority_letter,
            "gold_letter": sample["gold_letter"],
            "gold_answer": sample["gold_answer"],
            "category": sample["category"],
            "image_source": sample["image_source"],
            "is_correct": is_correct,
            "num_extracted": num_extracted,
            "num_samples": num_samples_sc,
            "letter_counts": letter_counts,
            "raw_outputs": raw_outputs if num_samples_sc <= 3 else raw_outputs[:1],
        })
    return batch_results


def evaluate_model(
    llm: LLM,
    processor: AutoProcessor,
    dataset: POPEDataset,
    model_name: str,
    args,
    lora_request=None,
) -> Dict[str, Any]:
    """Evaluate a model on the POPE dataset with batched vLLM generation."""
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
    batch_size = args.batch_size

    print(f"Evaluating {num_eval} samples: indices [{start_idx}, {end_idx})")
    print(f"Batch size: {batch_size}  |  Self-consistency samples: {args.num_samples_sc}")

    if args.num_samples_sc == 1:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.num_samples_sc,
            max_tokens=args.max_tokens,
        )

    num_batches = (num_eval + batch_size - 1) // batch_size
    batch_iter = range(num_batches)
    if tqdm:
        batch_iter = tqdm(batch_iter, desc=model_name, unit="batch")

    for batch_num in batch_iter:
        b_start = batch_num * batch_size
        b_end = min(b_start + batch_size, num_eval)
        batch_indices = eval_indices[b_start:b_end]

        batch_samples = []
        batch_inputs = []
        for idx in batch_indices:
            sample = dataset[idx]
            batch_samples.append(sample)

            question = make_text_prompt(sample["question"], sample["answer_choices"])
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": question},
                ],
            }]
            batch_inputs.append(prepare_inputs_for_vllm(messages, processor))

        if lora_request is not None:
            batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params, lora_request=lora_request)
        else:
            batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        batch_results = _parse_batch_outputs(batch_samples, batch_outputs, args.num_samples_sc)

        for result in batch_results:
            results.append(result)

            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}
            category_stats[cat]["total"] += 1

            if result["is_correct"]:
                correct += 1
                category_stats[cat]["correct"] += 1

        if not tqdm:
            acc = correct / len(results) * 100 if results else 0
            print(f"  [batch {batch_num + 1}/{num_batches}] {len(results)}/{num_eval} done, accuracy: {acc:.2f}%")

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
# Engine lifecycle helpers (identical to evaluate_mmvp)
# -------------------------
def cleanup_vllm_engine(llm: LLM) -> None:
    """Properly cleanup vLLM engine and free GPU memory."""
    print("\n[cleanup] Freeing GPU memory...")
    del llm
    gc.collect()
    gc.collect()

    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        print(f"[cleanup] Note: distributed cleanup returned: {e}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[cleanup] GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def merge_lora_weights(base_model_path: str, lora_adapter_path: str, output_dir: str) -> str:
    """Merge LoRA adapter weights into the base model and save to *output_dir*."""
    print(f"\n[merge] Merging LoRA weights...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA adapter: {lora_adapter_path}")
    print(f"  Output: {output_dir}")

    print("[merge] Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("[merge] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("[merge] Merging weights...")
    model = model.merge_and_unload()

    print("[merge] Saving merged model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    for fname in [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "preprocessor_config.json", "chat_template.json",
    ]:
        src = Path(base_model_path) / fname
        if src.exists():
            shutil.copy(src, Path(output_dir) / fname)

    del model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[merge] Done. Merged model saved to: {output_dir}")
    return output_dir


def init_vllm_engine(model_path: str, args, enable_lora: bool = False) -> LLM:
    """Initialize vLLM engine."""
    llm_kwargs: Dict[str, Any] = dict(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=True,
        dtype="auto",
        trust_remote_code=True,
        disable_log_stats=True,
    )

    if enable_lora:
        llm_kwargs["enable_lora"] = True

    while True:
        try:
            return LLM(**llm_kwargs)
        except TypeError as e:
            err_str = str(e)
            m = re.search(r"unexpected keyword argument '(\w+)'", err_str)
            if m:
                bad_key = m.group(1)
                print(f"[vLLM] Removing unsupported kwarg: {bad_key}")
                llm_kwargs.pop(bad_key, None)
            else:
                raise


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate models on POPE benchmark")

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
    parser.add_argument("--num_samples_sc", type=int, default=15,
                        help="Number of self-consistency samples per question (1=greedy)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (ignored if num_samples_sc=1)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of prompts to submit to vLLM in a single generate() call")
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)

    # Output
    parser.add_argument("--output_json", type=str, default="pope_results.json",
                        help="Output JSON file for results")

    # Model selection
    parser.add_argument("--eval_base", action="store_true", default=True,
                        help="Evaluate base model")
    parser.add_argument("--no_eval_base", action="store_false", dest="eval_base",
                        help="Skip base model evaluation")

    # LoRA merging
    parser.add_argument("--merge_weights", action="store_true", default=False,
                        help="Merge LoRA weights into base model before inference")

    args = parser.parse_args()

    base_snapshot = resolve_hf_snapshot_dir(args.base_model_dir)

    print("=" * 80)
    print("POPE BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Base model: {base_snapshot}")
    print(f"LoRA (VQA+Attn): {args.lora_vqa_attn_dir or 'None'}")
    print(f"LoRA (VQA only): {args.lora_vqa_only_dir or 'None'}")
    print(f"Merge LoRA weights: {args.merge_weights}")
    print(f"POPE cache dir: {args.pope_cache_dir}")
    print(f"Eval range: [{args.start_idx}, {args.end_idx})")
    print(f"Self-consistency samples: {args.num_samples_sc}")
    print(f"Batch size: {args.batch_size}")
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
            "num_samples_sc": args.num_samples_sc,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "merge_weights": args.merge_weights,
        },
        "models": {},
    }

    has_lora = (args.lora_vqa_attn_dir is not None) or (args.lora_vqa_only_dir is not None)
    temp_merged_dirs: List[str] = []

    # ---- LoRA models ----
    if has_lora:
        if args.merge_weights:
            print("\n" + "=" * 80)
            print("Merging LoRA weights into base model (--merge_weights enabled)")
            print("=" * 80)

            if args.lora_vqa_attn_dir is not None:
                merged_dir = tempfile.mkdtemp(prefix="merged_vqa_attn_")
                temp_merged_dirs.append(merged_dir)
                merge_lora_weights(base_snapshot, args.lora_vqa_attn_dir, merged_dir)

                print("\n[vLLM] Loading merged VQA+Attn model...")
                merged_llm = init_vllm_engine(merged_dir, args, enable_lora=False)

                result = evaluate_model(
                    llm=merged_llm, processor=processor, dataset=dataset,
                    model_name="VQA + Attention Alignment (merged)", args=args,
                )
                all_results["models"]["vqa_attn"] = result
                cleanup_vllm_engine(merged_llm)

            if args.lora_vqa_only_dir is not None:
                merged_dir = tempfile.mkdtemp(prefix="merged_vqa_only_")
                temp_merged_dirs.append(merged_dir)
                merge_lora_weights(base_snapshot, args.lora_vqa_only_dir, merged_dir)

                print("\n[vLLM] Loading merged VQA-only model...")
                merged_llm = init_vllm_engine(merged_dir, args, enable_lora=False)

                result = evaluate_model(
                    llm=merged_llm, processor=processor, dataset=dataset,
                    model_name="VQA Only (merged)", args=args,
                )
                all_results["models"]["vqa_only"] = result
                cleanup_vllm_engine(merged_llm)
        else:
            print("\n" + "=" * 80)
            print("Loading vLLM engine with LoRA support...")
            print("=" * 80)

            lora_llm = init_vllm_engine(base_snapshot, args, enable_lora=True)
            from vllm.lora.request import LoRARequest

            if args.lora_vqa_attn_dir is not None:
                lora_request = LoRARequest("vqa_attn", 1, args.lora_vqa_attn_dir)
                result = evaluate_model(
                    llm=lora_llm, processor=processor, dataset=dataset,
                    model_name="VQA + Attention Alignment", args=args,
                    lora_request=lora_request,
                )
                all_results["models"]["vqa_attn"] = result

            if args.lora_vqa_only_dir is not None:
                lora_request = LoRARequest("vqa_only", 2, args.lora_vqa_only_dir)
                result = evaluate_model(
                    llm=lora_llm, processor=processor, dataset=dataset,
                    model_name="VQA Only", args=args,
                    lora_request=lora_request,
                )
                all_results["models"]["vqa_only"] = result

            cleanup_vllm_engine(lora_llm)

    # ---- Base model ----
    if args.eval_base:
        print("\n" + "=" * 80)
        print("Loading vLLM engine for base model...")
        print("=" * 80)

        base_llm = init_vllm_engine(base_snapshot, args, enable_lora=False)

        result = evaluate_model(
            llm=base_llm, processor=processor, dataset=dataset,
            model_name="Base Qwen3-VL-8B", args=args,
        )
        all_results["models"]["base"] = result
        cleanup_vllm_engine(base_llm)

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

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] Results saved to {args.output_json}")

    # ---- Temp dir cleanup ----
    if temp_merged_dirs:
        print("\n[cleanup] Removing temporary merged model directories...")
        for temp_dir in temp_merged_dirs:
            try:
                shutil.rmtree(temp_dir)
                print(f"  Removed: {temp_dir}")
            except Exception as e:
                print(f"  Warning: Could not remove {temp_dir}: {e}")


if __name__ == "__main__":
    main()
