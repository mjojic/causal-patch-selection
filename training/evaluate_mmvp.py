#!/usr/bin/env python3
"""
Evaluate models on MMVP benchmark dataset.

Compares 3 model configurations:
1. Base Qwen 3 VL 8B
2. Fine-tuned with VQA prediction loss + attention alignment loss
3. Fine-tuned with VQA prediction loss only

Uses vLLM for efficient inference with optional self-consistency.
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

# Set other environment variables BEFORE any CUDA-related imports
os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Now safe to import everything else
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

# Import MMVP dataset loader
from mmvp_dataset_loader import MMVPDataset


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
# Prompting and parsing
# -------------------------
def make_text_prompt(question: str, answer_choices: List[str]) -> str:
    """Format question as multiple choice prompt for MMVP."""
    prompt = (
        "Given the following question and answer choices about an image, do the following:\n"
        "1) Look carefully at the image and identify visual details relevant to the question.\n"
        "2) Reason about each answer choice based on what you observe.\n"
        "3) Select the single best answer choice.\n\n"
        f"Question: {question}\n"
        "Answer Choices:\n"
    )
    options = ["A", "B", "C", "D", "E"]
    for i, choice in enumerate(answer_choices):
        prompt += f"{options[i]}: {choice}\n"
    prompt += "\nProvide your final answer in this JSON format: {'final_answer_letter': <letter>}\n"
    return prompt


def parse_model_output(text: str) -> Optional[str]:
    """Extract answer letter from model output."""
    # Try JSON format
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

    # Try key-value format
    m2 = re.search(r"final_answer_letter\s*[:=]\s*['\"]?([A-Ea-e])['\"]?", text)
    if m2:
        return m2.group(1).upper()

    # Try standalone letter at end
    m3 = re.search(r"^\s*([A-E])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()
    
    # Try "Answer: X" or "The answer is X" patterns
    m4 = re.search(r"(?:answer|choice)(?:\s+is)?[:\s]+([A-E])\b", text, re.IGNORECASE)
    if m4:
        return m4.group(1).upper()

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
def evaluate_sample(
    llm: LLM,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    lora_request=None,
) -> Dict[str, Any]:
    """Evaluate a single MMVP sample with optional self-consistency."""
    question = make_text_prompt(sample["question"], sample["answer_choices"])
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": question}
        ],
    }]
    
    vllm_input = prepare_inputs_for_vllm(messages, processor)
    
    # Use temperature=0 for single sample, otherwise use provided temperature
    if num_samples_sc == 1:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            n=num_samples_sc,
            max_tokens=max_tokens,
        )
    
    # Generate with optional LoRA
    if lora_request is not None:
        outputs = llm.generate([vllm_input], sampling_params=sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    
    # Parse outputs
    num_extracted = 0
    letter_counts: Dict[str, int] = {}
    raw_outputs = []
    
    for gen in outputs[0].outputs:
        raw_outputs.append(gen.text)
        parsed = parse_model_output(gen.text)
        if parsed:
            num_extracted += 1
            letter_counts[parsed] = letter_counts.get(parsed, 0) + 1
    
    # Determine majority answer
    majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
    
    # Check correctness
    is_correct = (majority_letter == sample["gold_letter"]) if majority_letter else False
    
    return {
        "index": sample["index"],
        "question": sample["question"],
        "answer_choices": sample["answer_choices"],
        "predicted_letter": majority_letter,
        "gold_letter": sample["gold_letter"],
        "gold_answer": sample["gold_answer"],
        "is_correct": is_correct,
        "num_extracted": num_extracted,
        "num_samples": num_samples_sc,
        "letter_counts": letter_counts,
        "raw_outputs": raw_outputs if num_samples_sc <= 3 else raw_outputs[:1],  # Save space
    }


def evaluate_model(
    llm: LLM,
    processor: AutoProcessor,
    dataset: MMVPDataset,
    model_name: str,
    args,
    lora_request=None,
) -> Dict[str, Any]:
    """Evaluate a model on the MMVP dataset."""
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 80}")
    
    results = []
    correct = 0
    
    # Determine indices to evaluate
    start_idx = args.start_idx
    end_idx = min(args.end_idx, len(dataset))
    eval_indices = list(range(start_idx, end_idx))
    num_eval = len(eval_indices)
    
    print(f"Evaluating {num_eval} samples: indices [{start_idx}, {end_idx})")
    
    iterator = tqdm(eval_indices, desc=model_name) if tqdm else eval_indices
    
    for idx in iterator:
        sample = dataset[idx]
        
        result = evaluate_sample(
            llm=llm,
            processor=processor,
            sample=sample,
            num_samples_sc=args.num_samples_sc,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            lora_request=lora_request,
        )
        
        results.append(result)
        if result["is_correct"]:
            correct += 1
        
        # Progress update
        if not tqdm and len(results) % 25 == 0:
            acc = correct / len(results) * 100
            print(f"  [{len(results)}/{num_eval}] Current accuracy: {acc:.2f}%")
    
    accuracy = correct / num_eval * 100
    print(f"\n[{model_name}] Final accuracy: {accuracy:.2f}% ({correct}/{num_eval})")
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": num_eval,
        "results": results,
    }


def cleanup_vllm_engine(llm: LLM) -> None:
    """Properly cleanup vLLM engine and free GPU memory."""
    print("\n[cleanup] Freeing GPU memory...")
    
    # Delete the LLM object
    del llm
    
    # Force garbage collection multiple times to ensure all references are cleaned up
    gc.collect()
    gc.collect()
    
    # Destroy vLLM distributed state (important for releasing GPU memory)
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        print(f"[cleanup] Note: distributed cleanup returned: {e}")
    
    # Synchronize CUDA operations before clearing cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Additional garbage collection after CUDA cleanup
    gc.collect()
    
    # Log memory status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[cleanup] GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def merge_lora_weights(base_model_path: str, lora_adapter_path: str, output_dir: str) -> str:
    """
    Merge LoRA adapter weights into the base model and save to output directory.
    
    Args:
        base_model_path: Path to base model snapshot
        lora_adapter_path: Path to LoRA adapter
        output_dir: Directory to save merged model
        
    Returns:
        Path to merged model directory
    """
    print(f"\n[merge] Merging LoRA weights...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA adapter: {lora_adapter_path}")
    print(f"  Output: {output_dir}")
    
    # Load base model in bfloat16 for merging
    print("[merge] Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU to avoid OOM during merge
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapter
    print("[merge] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Merge and unload
    print("[merge] Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    print("[merge] Saving merged model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # Copy processor/tokenizer files from base model
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", 
                  "special_tokens_map.json", "preprocessor_config.json", "chat_template.json"]:
        src = Path(base_model_path) / fname
        if src.exists():
            shutil.copy(src, Path(output_dir) / fname)
    
    # Clean up to free memory
    del model
    del base_model
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
    
    # Try to create engine, removing unsupported kwargs if necessary
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
    parser = argparse.ArgumentParser(description="Evaluate models on MMVP benchmark")
    
    # Model paths
    parser.add_argument("--base_model_dir", type=str, 
                        default="/mnt/shared/shared_hf_home /models--Qwen--Qwen3-VL-8B-Instruct",
                        help="Path to base model directory")
    parser.add_argument("--lora_vqa_attn_dir", type=str, default=None,
                        help="Path to LoRA adapter trained with VQA + attention alignment loss")
    parser.add_argument("--lora_vqa_only_dir", type=str, default=None,
                        help="Path to LoRA adapter trained with VQA loss only")
    
    # Dataset
    parser.add_argument("--mmvp_dir", type=str, default="/mnt/arc/mjojic/data/MMVP",
                        help="Path to MMVP dataset directory")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in dataset (inclusive)")
    parser.add_argument("--end_idx", type=int, default=300,
                        help="End index in dataset (exclusive)")
    
    # Inference settings
    parser.add_argument("--num_samples_sc", type=int, default=15,
                        help="Number of self-consistency samples per question (1=greedy)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (ignored if num_samples_sc=1)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hardware
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    
    # Output
    parser.add_argument("--output_json", type=str, default="mmvp_results.json",
                        help="Output JSON file for results")
    
    # Model selection (which models to evaluate)
    parser.add_argument("--eval_base", action="store_true", default=True,
                        help="Evaluate base model")
    parser.add_argument("--no_eval_base", action="store_false", dest="eval_base",
                        help="Skip base model evaluation")
    
    # LoRA merging
    parser.add_argument("--merge_weights", action="store_true", default=False,
                        help="Merge LoRA weights into base model before inference (faster but uses more disk space)")
    
    args = parser.parse_args()
    
    # Note: CUDA_VISIBLE_DEVICES is set at module load time via _parse_gpu_early()
    # to ensure it's set before torch/CUDA initialization
    
    # Resolve model path
    base_snapshot = resolve_hf_snapshot_dir(args.base_model_dir)
    
    print("=" * 80)
    print("MMVP BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Base model: {base_snapshot}")
    print(f"LoRA (VQA+Attn): {args.lora_vqa_attn_dir or 'None'}")
    print(f"LoRA (VQA only): {args.lora_vqa_only_dir or 'None'}")
    print(f"Merge LoRA weights: {args.merge_weights}")
    print(f"MMVP dataset: {args.mmvp_dir}")
    print(f"Eval range: [{args.start_idx}, {args.end_idx})")
    print(f"Self-consistency samples: {args.num_samples_sc}")
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
            repo_id, trust_remote_code=True, local_files_only=True, cache_dir=hub_cache
        )
    
    # Load MMVP dataset
    print("\n[dataset] Loading MMVP...")
    dataset = MMVPDataset(mmvp_dir=args.mmvp_dir)
    
    # Adjust end_idx if necessary
    if args.end_idx > len(dataset):
        print(f"[warning] end_idx={args.end_idx} > dataset size={len(dataset)}, adjusting...")
        args.end_idx = len(dataset)
    
    # Store all results
    all_results = {
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
            "merge_weights": args.merge_weights,
        },
        "models": {},
    }
    
    # Determine which LoRA adapters to evaluate
    has_lora = (args.lora_vqa_attn_dir is not None) or (args.lora_vqa_only_dir is not None)
    
    # Track temporary merged model directories for cleanup
    temp_merged_dirs = []
    
    # If we have LoRA adapters, evaluate them
    if has_lora:
        if args.merge_weights:
            # Merge LoRA weights into base model for faster inference
            print("\n" + "=" * 80)
            print("Merging LoRA weights into base model (--merge_weights enabled)")
            print("=" * 80)
            
            # Evaluate VQA + Attention alignment model (merged)
            if args.lora_vqa_attn_dir is not None:
                # Create temp directory for merged model
                merged_dir = tempfile.mkdtemp(prefix="merged_vqa_attn_")
                temp_merged_dirs.append(merged_dir)
                
                merge_lora_weights(base_snapshot, args.lora_vqa_attn_dir, merged_dir)
                
                print("\n[vLLM] Loading merged VQA+Attn model...")
                merged_llm = init_vllm_engine(merged_dir, args, enable_lora=False)
                
                result = evaluate_model(
                    llm=merged_llm,
                    processor=processor,
                    dataset=dataset,
                    model_name="VQA + Attention Alignment (merged)",
                    args=args,
                    lora_request=None,
                )
                all_results["models"]["vqa_attn"] = result
                
                cleanup_vllm_engine(merged_llm)
            
            # Evaluate VQA only model (merged)
            if args.lora_vqa_only_dir is not None:
                # Create temp directory for merged model
                merged_dir = tempfile.mkdtemp(prefix="merged_vqa_only_")
                temp_merged_dirs.append(merged_dir)
                
                merge_lora_weights(base_snapshot, args.lora_vqa_only_dir, merged_dir)
                
                print("\n[vLLM] Loading merged VQA-only model...")
                merged_llm = init_vllm_engine(merged_dir, args, enable_lora=False)
                
                result = evaluate_model(
                    llm=merged_llm,
                    processor=processor,
                    dataset=dataset,
                    model_name="VQA Only (merged)",
                    args=args,
                    lora_request=None,
                )
                all_results["models"]["vqa_only"] = result
                
                cleanup_vllm_engine(merged_llm)
        else:
            # Use vLLM's native LoRA support (original behavior)
            print("\n" + "=" * 80)
            print("Loading vLLM engine with LoRA support...")
            print("=" * 80)
            
            lora_llm = init_vllm_engine(base_snapshot, args, enable_lora=True)
            
            # Import LoRA request
            from vllm.lora.request import LoRARequest
            
            # Evaluate VQA + Attention alignment model
            if args.lora_vqa_attn_dir is not None:
                lora_request = LoRARequest("vqa_attn", 1, args.lora_vqa_attn_dir)
                result = evaluate_model(
                    llm=lora_llm,
                    processor=processor,
                    dataset=dataset,
                    model_name="VQA + Attention Alignment",
                    args=args,
                    lora_request=lora_request,
                )
                all_results["models"]["vqa_attn"] = result
            
            # Evaluate VQA only model
            if args.lora_vqa_only_dir is not None:
                lora_request = LoRARequest("vqa_only", 2, args.lora_vqa_only_dir)
                result = evaluate_model(
                    llm=lora_llm,
                    processor=processor,
                    dataset=dataset,
                    model_name="VQA Only",
                    args=args,
                    lora_request=lora_request,
                )
                all_results["models"]["vqa_only"] = result
            
            # Clean up LoRA engine thoroughly before loading base model
            cleanup_vllm_engine(lora_llm)
    
    # Evaluate base model
    if args.eval_base:
        
        print("\n" + "=" * 80)
        print("Loading vLLM engine for base model...")
        print("=" * 80)
        
        base_llm = init_vllm_engine(base_snapshot, args, enable_lora=False)
        
        result = evaluate_model(
            llm=base_llm,
            processor=processor,
            dataset=dataset,
            model_name="Base Qwen3-VL-8B",
            args=args,
            lora_request=None,
        )
        all_results["models"]["base"] = result
        
        # Clean up
        cleanup_vllm_engine(base_llm)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for model_key, model_result in all_results["models"].items():
        name = model_result["model_name"]
        acc = model_result["accuracy"]
        correct = model_result["correct"]
        total = model_result["total"]
        print(f"  {name:30s}: {acc:5.2f}% ({correct}/{total})")
    
    # Calculate improvements if we have multiple models
    if "base" in all_results["models"]:
        base_acc = all_results["models"]["base"]["accuracy"]
        for model_key in ["vqa_attn", "vqa_only"]:
            if model_key in all_results["models"]:
                other_acc = all_results["models"][model_key]["accuracy"]
                diff = other_acc - base_acc
                print(f"  {all_results['models'][model_key]['model_name']} vs Base: {diff:+.2f}%")
    
    print("=" * 80)
    
    # Save results
    # Convert results to JSON-serializable format (remove PIL images from raw outputs)
    for model_key in all_results["models"]:
        for r in all_results["models"][model_key]["results"]:
            # Remove any non-serializable data
            r.pop("image", None)
    
    with open(args.output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] Results saved to {args.output_json}")
    
    # Clean up temporary merged model directories
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
