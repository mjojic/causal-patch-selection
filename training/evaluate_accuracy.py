#!/usr/bin/env python3
"""
Evaluate accuracy of base vs fine-tuned model on NaturalBench dataset.
Uses self-consistency to determine final predictions.
"""

import os
import sys
import argparse

# ----------------------------
# CRITICAL: Set environment variables BEFORE any CUDA-related imports
# ----------------------------
os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Now safe to import everything else
import re
import json
import glob
import shutil
import tempfile
import multiprocessing
import gc
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import datasets
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from peft import PeftModel


# -------------------------
# Image resizing (consistent with training)
# -------------------------
MAX_IMAGE_SIDE = 1024  # Resize images so largest side is at most this


def _resample_lanczos():
    """Pillow compatibility for LANCZOS resampling."""
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def maybe_downscale_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """
    Downscale image if its largest side exceeds max_side.
    Preserves aspect ratio. Returns original if already small enough.
    """
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

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# -------------------------
# Environment
# -------------------------
os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_TORCH_COMPILE", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


# -------------------------
# Model path resolution (from search.py and attention_alignment.py)
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
# Dataset loading (from search.py)
# -------------------------
def load_naturalbench_dataset(split: str):
    return datasets.load_dataset("BaiqiL/NaturalBench-lmms-eval", split=split)


def datapoint_to_dict(dp):
    """Convert dataset point to standardized format"""
    img = dp["Image"]
    # Downscale image if largest side > 1024 (consistent with training)
    img = maybe_downscale_image(img.convert("RGB"))
    question_raw = dp["Question"]
    q_type = dp["Question_Type"]
    ans_raw = str(dp["Answer"]).strip()

    base_question = question_raw
    answer_choices: List[str] = []
    gold_letter: Optional[str] = None
    gold_answer: Optional[str] = None

    if q_type == "yes_no":
        answer_choices = ["Yes", "No"]
        if ans_raw.lower().startswith("y"):
            gold_letter = "A"
            gold_answer = "Yes"
        else:
            gold_letter = "B"
            gold_answer = "No"
    else:
        optA, optB = "Option A", "Option B"
        if "Option:" in question_raw:
            head, tail = question_raw.split("Option:", 1)
            base_question = head.strip()
            segs = [s.strip() for s in tail.split(";") if s.strip()]
            parsed = {}
            for s in segs:
                if ":" in s:
                    key, val = s.split(":", 1)
                    key = key.strip().upper()
                    parsed[key] = val.strip()
            optA = parsed.get("A", optA)
            optB = parsed.get("B", optB)
        else:
            m2 = re.search(r"\bA\.\s*(.*?)\s*B\.\s*(.*)$", question_raw)
            if m2:
                base_question = question_raw[:m2.start()].strip()
                optA = m2.group(1).strip()
                optB = m2.group(2).strip()

        answer_choices = [optA, optB]
        gold_letter = ans_raw.upper()
        idx = ord(gold_letter) - ord("A")
        gold_answer = answer_choices[idx] if 0 <= idx < len(answer_choices) else ans_raw

    return {
        "image": img,
        "question": base_question,
        "answer_choices": answer_choices,
        "gold_answer": gold_answer,
        "gold_letter": gold_letter,
        "spurious_attribute": q_type,
    }


# -------------------------
# Prompting and parsing (from search.py)
# -------------------------
def make_text_prompt(question: str, answer_choices: List[str]) -> str:
    """Format question as multiple choice prompt"""
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
    """Extract answer letter from model output"""
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

    # Try standalone letter
    m3 = re.search(r"^\s*([A-E])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()

    return None


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    """Prepare inputs for vLLM inference"""
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


def evaluate_single_sample(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run self-consistency evaluation on a single sample"""
    question = make_text_prompt(data_dict["question"], data_dict["answer_choices"])
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": data_dict["image"]},
            {"type": "text", "text": question}
        ],
    }]
    
    vllm_input = prepare_inputs_for_vllm(messages, processor)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples_sc,
        max_tokens=max_tokens,
    )
    
    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    
    # Parse outputs
    num_extracted = 0
    letter_counts: Dict[str, int] = {}
    
    for gen in outputs[0].outputs:
        parsed = parse_qwen_output(gen.text)
        if parsed:
            num_extracted += 1
            letter_counts[parsed] = letter_counts.get(parsed, 0) + 1
    
    # Determine majority answer
    majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
    
    # Check correctness
    is_correct = (majority_letter == data_dict["gold_letter"]) if majority_letter else False
    
    return {
        "predicted_letter": majority_letter,
        "gold_letter": data_dict["gold_letter"],
        "is_correct": is_correct,
        "num_extracted": num_extracted,
        "letter_counts": letter_counts,
    }


# -------------------------
# LoRA weight merging
# -------------------------
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


# -------------------------
# vLLM initialization (from search.py)
# -------------------------
def init_vllm_engine(model_path: str, repo_id: str, hub_cache: str, args) -> LLM:
    """Initialize vLLM engine"""
    lower = model_path.lower()
    is_qwen25 = ("qwen2.5" in lower) or ("qwen25" in lower)

    llm_kwargs: Dict[str, Any] = dict(
        model=model_path,  # Use local path to avoid HF hub access
        tokenizer=model_path,  # Use local path to avoid HF hub access
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=True,
        dtype="auto",
        trust_remote_code=True,
        disable_log_stats=True,
    )

    if is_qwen25:
        llm_kwargs["quantization"] = "fp8"
        llm_kwargs["kv_cache_dtype"] = "fp8"
        llm_kwargs["calculate_kv_scales"] = True

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
# Main evaluation
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, required=True,
                       help="Path to base model directory")
    parser.add_argument("--lora_dir", type=str, required=True,
                       help="Path to LoRA adapter directory")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index in dataset (inclusive)")
    parser.add_argument("--end_idx", type=int, default=400,
                       help="End index in dataset (exclusive)")
    parser.add_argument("--num_samples_sc", type=int, default=15,
                       help="Number of self-consistency samples per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=17000)
    parser.add_argument("--output_json", type=str, default="accuracy_results.json")
    parser.add_argument("--merge_weights", action="store_true", default=False,
                        help="Merge LoRA weights into base model before inference (faster but uses more disk space)")
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Resolve model paths
    base_snapshot = resolve_hf_snapshot_dir(args.base_model_dir)
    base_repo_id = _guess_repo_id_from_models_dir(args.base_model_dir)
    hub_cache = _shared_hf_hub_cache(args.base_model_dir)
    
    print("=" * 80)
    print("ACCURACY EVALUATION: Base vs Fine-tuned Model")
    print("=" * 80)
    print(f"Base model: {base_snapshot}")
    print(f"LoRA adapters: {args.lora_dir}")
    print(f"Merge LoRA weights: {args.merge_weights}")
    print(f"Evaluating indices [{args.start_idx}, {args.end_idx}) with {args.num_samples_sc} SC samples each")
    print("=" * 80)
    
    # Load processor
    print("\n[processor] Loading...")
    try:
        processor = AutoProcessor.from_pretrained(base_snapshot, trust_remote_code=True, local_files_only=True)
    except OSError:
        print(f"[processor] Fallback to repo_id={base_repo_id}")
        processor = AutoProcessor.from_pretrained(
            base_repo_id, trust_remote_code=True, local_files_only=True, cache_dir=hub_cache
        )
    
    # Load dataset
    print("\n[dataset] Loading NaturalBench test split...")
    ds = load_naturalbench_dataset(split="test")
    print(f"[dataset] Total samples: {len(ds)}")
    
    # Validate and adjust indices
    start_idx = max(0, args.start_idx)
    end_idx = min(args.end_idx, len(ds))
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx} >= end_idx={end_idx}")
    
    eval_indices = list(range(start_idx, end_idx))
    num_eval = len(eval_indices)
    print(f"[dataset] Evaluating {num_eval} samples: indices [{start_idx}, {end_idx})")
    
    # Track temporary merged model directories for cleanup
    temp_merged_dirs = []
    
    # Initialize fine-tuned model
    print("\n" + "=" * 80)
    print("Loading FINE-TUNED model...")
    print("=" * 80)
    
    lora_request = None  # Will be set if using vLLM LoRA support
    
    if args.merge_weights:
        # Merge LoRA weights into base model for faster inference
        print("Merging LoRA weights into base model (--merge_weights enabled)")
        
        # Create temp directory for merged model
        merged_dir = tempfile.mkdtemp(prefix="merged_finetuned_")
        temp_merged_dirs.append(merged_dir)
        
        merge_lora_weights(base_snapshot, args.lora_dir, merged_dir)
        
        # Load merged model (no LoRA needed)
        llm_kwargs_tuned = dict(
            model=merged_dir,
            tokenizer=merged_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            enforce_eager=True,
            dtype="auto",
            trust_remote_code=True,
            disable_log_stats=True,
        )
    else:
        # Use vLLM's native LoRA support
        # For fine-tuned model, we need to load base + adapters
        # vLLM supports LoRA adapters via enable_lora=True and lora_modules
        llm_kwargs_tuned = dict(
            model=base_snapshot,  # Use local path to avoid HF hub access
            tokenizer=base_snapshot,  # Use local path to avoid HF hub access
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            enforce_eager=True,
            dtype="auto",
            trust_remote_code=True,
            disable_log_stats=True,
            enable_lora=True,
        )
        
        # Import LoRA request for later use
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("finetuned", 1, args.lora_dir)
    
    # Try to create engine
    while True:
        try:
            tuned_llm = LLM(**llm_kwargs_tuned)
            break
        except TypeError as e:
            err_str = str(e)
            m = re.search(r"unexpected keyword argument '(\w+)'", err_str)
            if m:
                bad_key = m.group(1)
                print(f"[vLLM] Removing unsupported kwarg: {bad_key}")
                llm_kwargs_tuned.pop(bad_key, None)
            else:
                raise
    
    # Evaluate fine-tuned model
    print("\n" + "=" * 80)
    print("Evaluating FINE-TUNED model...")
    print("=" * 80)
    tuned_correct = 0
    tuned_results = []
    
    iterator = tqdm(eval_indices, desc="Fine-tuned model") if tqdm else eval_indices
    for idx in iterator:
        data_dict = datapoint_to_dict(ds[idx])
        
        # Prepare input
        question = make_text_prompt(data_dict["question"], data_dict["answer_choices"])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": data_dict["image"]},
                {"type": "text", "text": question}
            ],
        }]
        vllm_input = prepare_inputs_for_vllm(messages, processor)
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.num_samples_sc,
            max_tokens=args.max_tokens,
        )
        
        # Generate (with or without LoRA adapter depending on merge_weights)
        if lora_request is not None:
            outputs = tuned_llm.generate(
                [vllm_input],
                sampling_params=sampling_params,
                lora_request=lora_request
            )
        else:
            outputs = tuned_llm.generate(
                [vllm_input],
                sampling_params=sampling_params,
            )
        
        # Parse outputs
        num_extracted = 0
        letter_counts: Dict[str, int] = {}
        
        for gen in outputs[0].outputs:
            parsed = parse_qwen_output(gen.text)
            if parsed:
                num_extracted += 1
                letter_counts[parsed] = letter_counts.get(parsed, 0) + 1
        
        majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
        is_correct = (majority_letter == data_dict["gold_letter"]) if majority_letter else False
        
        result = {
            "predicted_letter": majority_letter,
            "gold_letter": data_dict["gold_letter"],
            "is_correct": is_correct,
            "num_extracted": num_extracted,
            "letter_counts": letter_counts,
        }
        tuned_results.append(result)
        if is_correct:
            tuned_correct += 1
        
        if not tqdm and len(tuned_results) % 50 == 0:
            acc = tuned_correct / len(tuned_results) * 100
            print(f"[{len(tuned_results)}/{num_eval}] (idx {idx}) Tuned accuracy: {acc:.2f}%")
    
    tuned_accuracy = tuned_correct / num_eval * 100
    print(f"\n[FINE-TUNED] Final accuracy: {tuned_accuracy:.2f}% ({tuned_correct}/{num_eval})")
    
    del tuned_llm
    torch.cuda.empty_cache()


    # Initialize base model
    print("\n" + "=" * 80)
    print("Loading BASE model...")
    print("=" * 80)
    base_llm = init_vllm_engine(base_snapshot, base_repo_id, hub_cache, args)
    
    # Evaluate base model
    print("\n" + "=" * 80)
    print("Evaluating BASE model...")
    print("=" * 80)
    base_correct = 0
    base_results = []
    
    iterator = tqdm(eval_indices, desc="Base model") if tqdm else eval_indices
    for idx in iterator:
        data_dict = datapoint_to_dict(ds[idx])
        result = evaluate_single_sample(
            base_llm, processor, data_dict,
            args.num_samples_sc, args.temperature, args.top_p, args.max_tokens
        )
        base_results.append(result)
        if result["is_correct"]:
            base_correct += 1
        
        if not tqdm and len(base_results) % 50 == 0:
            acc = base_correct / len(base_results) * 100
            print(f"[{len(base_results)}/{num_eval}] (idx {idx}) Base accuracy: {acc:.2f}%")
    
    base_accuracy = base_correct / num_eval * 100
    print(f"\n[BASE] Final accuracy: {base_accuracy:.2f}% ({base_correct}/{num_eval})")
    
    # Clean up base model
    del base_llm
    torch.cuda.empty_cache()
    

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Base model accuracy:       {base_accuracy:.2f}% ({base_correct}/{num_eval})")
    print(f"Fine-tuned model accuracy: {tuned_accuracy:.2f}% ({tuned_correct}/{num_eval})")
    improvement = tuned_accuracy - base_accuracy
    print(f"Improvement:               {improvement:+.2f}%")
    print("=" * 80)
    
    # Save results
    results = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_samples": num_eval,
        "merge_weights": args.merge_weights,
        "base_accuracy": base_accuracy,
        "base_correct": base_correct,
        "tuned_accuracy": tuned_accuracy,
        "tuned_correct": tuned_correct,
        "improvement": improvement,
        "base_results": base_results,
        "tuned_results": tuned_results,
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
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
