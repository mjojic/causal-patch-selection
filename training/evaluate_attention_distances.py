import os
import math
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from peft import PeftModel
from qwen_vl_utils import process_vision_info
from consensus_mask_dataloader import ConsensusMaskDataset

# -------------------------
# Env
# -------------------------
os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Import helper functions from attention_alignment
from attention_alignment import (
    resolve_hf_snapshot_dir,
    _guess_repo_id_from_models_dir,
    _shared_hf_hub_cache,
    format_mc_prompt,
    build_inputs_for_batch,
    find_vision_spans,
    find_last_attention_module,
    get_head_meta,
    _unwrap_peft,
    get_letter_token_ids,
    find_answer_token_pos,
    forward_core_model_for_hidden_states,
)


def load_base_model(
    model_path: str,
    gpu_id: int,
    attn_implementation: str = "flash_attention_2",
    quantization: str = "4bit",
    bnb_compute_dtype: torch.dtype = torch.bfloat16,
):
    """Load base model without LoRA"""
    snapshot_path = resolve_hf_snapshot_dir(model_path)
    print(f"[base model] Resolved snapshot: {snapshot_path}")

    # processor
    try:
        processor = AutoProcessor.from_pretrained(
            snapshot_path, trust_remote_code=True, local_files_only=True
        )
    except (OSError, ValueError):
        repo_id = _guess_repo_id_from_models_dir(model_path)
        hub_cache = _shared_hf_hub_cache(model_path)
        print(f"[processor] fallback repo_id={repo_id} cache_dir={hub_cache}")
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=True, cache_dir=hub_cache
        )

    # Check if model is already quantized (e.g., FP8)
    import json
    config_path = Path(snapshot_path) / "config.json"
    is_prequantized = False
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
            if "quantization_config" in config_data:
                is_prequantized = True
                print(f"[base model] Model already quantized with {config_data['quantization_config'].get('quant_method', 'unknown')}")

    quantization = (quantization or "none").lower()
    load_kwargs = {}
    
    if is_prequantized:
        # Model already has quantization, don't add BitsAndBytes
        print("[base model] Using pre-quantized model weights.")
        load_kwargs["dtype"] = torch.bfloat16
    elif quantization != "none":
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig import failed.")
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )
            print("[base model] Using 4-bit quantization.")
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("[base model] Using 8-bit quantization.")
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu_id}"},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    model.config.use_cache = False
    return model, processor


def load_finetuned_model(
    base_model_path: str,
    adapter_path: str,
    gpu_id: int,
    attn_implementation: str = "flash_attention_2",
    quantization: str = "4bit",
    bnb_compute_dtype: torch.dtype = torch.bfloat16,
):
    """Load fine-tuned model with LoRA adapters"""
    # For LoRA models, we need to use BitsAndBytes quantization, not pre-quantized models
    # because FP8 dtype is incompatible with LoRA operations
    snapshot_path = resolve_hf_snapshot_dir(base_model_path)
    print(f"[finetuned model] Resolved snapshot: {snapshot_path}")

    # processor
    try:
        processor = AutoProcessor.from_pretrained(
            snapshot_path, trust_remote_code=True, local_files_only=True
        )
    except (OSError, ValueError):
        repo_id = _guess_repo_id_from_models_dir(base_model_path)
        hub_cache = _shared_hf_hub_cache(base_model_path)
        print(f"[processor] fallback repo_id={repo_id} cache_dir={hub_cache}")
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=True, cache_dir=hub_cache
        )

    # Always use BitsAndBytes quantization for LoRA compatibility
    quantization = (quantization or "none").lower()
    load_kwargs = {}
    
    if quantization == "4bit":
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig import failed.")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )
        print("[finetuned model] Using 4-bit BNB quantization (required for LoRA).")
    elif quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("[finetuned model] Using 8-bit BNB quantization (required for LoRA).")
    else:
        load_kwargs["dtype"] = torch.bfloat16
        print("[finetuned model] Using bfloat16 (no quantization).")

    base_model = AutoModelForImageTextToText.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu_id}"},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    base_model.config.use_cache = False
    
    print(f"[finetuned model] Loading adapters from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.config.use_cache = False
    return model, processor


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def collate_fn_factory(processor: AutoProcessor):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [b["image"].convert("RGB") for b in batch]
        prompts = [format_mc_prompt(b["question"], b["answer_choices"]) for b in batch]
        gold_letters = [b["gold_letter"] for b in batch]
        token_masks = [b["token_mask"] for b in batch]

        inputs = build_inputs_for_batch(processor, images, prompts, gold_letters)
        return {
            "inputs": inputs,
            "gold_letters": gold_letters,
            "token_masks": token_masks,
        }
    return collate


def compute_attention_distance_single_sample(
    model,
    inputs: Dict[str, torch.Tensor],
    gold_letter: str,
    token_mask_hw: torch.Tensor,
    processor,
    eps: float = 1e-8,
) -> float:
    """
    Compute attention distance for a single sample.
    Returns the Frobenius distance or None if computation fails.
    """
    base = _unwrap_peft(model)
    _, attn_mod = find_last_attention_module(base)

    captured = {}

    def pre_hook(mod, args, kwargs):
        if len(args) > 0 and torch.is_tensor(args[0]):
            hs = args[0]
        elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            hs = kwargs["hidden_states"]
        else:
            hs = None
            for v in kwargs.values():
                if torch.is_tensor(v):
                    hs = v
                    break
        captured["hs"] = hs
        return None

    h = attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        _core_out, _mode = forward_core_model_for_hidden_states(model, inputs)
    finally:
        h.remove()

    hs_all = captured.get("hs", None)
    if hs_all is None:
        return None

    device = hs_all.device
    B, S, D = hs_all.shape

    letters, letter_ids = get_letter_token_ids(processor.tokenizer)

    if gold_letter not in letters:
        return None
    
    y = letters.index(gold_letter)
    gold_tid = letter_ids[y]
    
    try:
        ans_pos = find_answer_token_pos(inputs["input_ids"][0], inputs["attention_mask"][0], gold_tid)
    except:
        return None

    spans = find_vision_spans(inputs["input_ids"][0], model)
    if not spans:
        return None
    
    vs, ve = spans[0]
    img_pos = torch.arange(vs + 1, ve, device=device)
    I = img_pos.numel()
    if I <= 0:
        return None

    m_hw = token_mask_hw.to(device=device, dtype=torch.bool)
    m = m_hw.flatten().to(dtype=torch.float32)
    if m.numel() != I:
        return None
    if m.sum() <= 0:
        return None

    q_raw = attn_mod.q_proj(hs_all[0:1, ans_pos:ans_pos+1, :])
    k_raw = attn_mod.k_proj(hs_all[0:1, img_pos, :])

    num_heads, num_kv_heads, head_dim = get_head_meta(attn_mod, base, q_raw, k_raw)

    q = q_raw.view(1, 1, num_heads, head_dim).transpose(1, 2)
    k = k_raw.view(1, I, num_kv_heads, head_dim).transpose(1, 2)

    rep = num_heads // num_kv_heads
    if rep != 1:
        k = k.repeat_interleave(rep, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    p = torch.softmax(scores, dim=-1).squeeze(2).mean(dim=1).squeeze(0)
    p = (p + eps) / (p.sum() + eps * I)

    t = (m + eps) / (m.sum() + eps * I)
    
    # Frobenius norm distance (squared)
    distance = ((p - t) ** 2).sum().item()
    
    return distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--base_model_path", type=str,
                        default="/mnt/arc/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct",
                        help="Path to base model (can be FP8 or regular)")
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="Path to fine-tuned model with LoRA adapters")
    parser.add_argument("--consensus_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="attention_distances.csv")
    parser.add_argument("--output_plot", type=str, default="attention_distances.png")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--quantization", type=str, default="4bit",
                        choices=["none", "4bit", "8bit"])
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print("=" * 80)
    print("Loading BASE model...")
    print("=" * 80)
    base_model, processor = load_base_model(
        args.base_model_path,
        gpu_id=args.gpu,
        attn_implementation=args.attn_implementation,
        quantization=args.quantization,
    )
    base_model.eval()

    print("\n" + "=" * 80)
    print("Loading FINE-TUNED model...")
    print("=" * 80)
    finetuned_model, _ = load_finetuned_model(
        args.base_model_path,
        args.finetuned_model_path,
        gpu_id=args.gpu,
        attn_implementation=args.attn_implementation,
        quantization=args.quantization,
    )
    finetuned_model.eval()

    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    dataset = ConsensusMaskDataset(
        consensus_dir=args.consensus_dir,
        model=base_model,
        processor=processor,
        precompute_token_masks=True,
        use_precomputed_masks=True,
        cache_images=False,
        cache_masks=False,
        max_side=1024,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_factory(processor),
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print("\n" + "=" * 80)
    print("Computing attention distances...")
    print("=" * 80)

    results = []
    sample_idx = 0  # Track dataset index manually
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating") if tqdm else loader):
            inputs = _to_device(batch["inputs"], device)
            gold_letters = batch["gold_letters"]
            token_masks = batch["token_masks"]

            for i in range(len(gold_letters)):
                idx = sample_idx
                
                # Create single-sample batch
                single_inputs = {
                    k: v[i:i+1] if torch.is_tensor(v) else v
                    for k, v in inputs.items()
                }
                
                # Compute distance for base model
                base_dist = compute_attention_distance_single_sample(
                    base_model,
                    single_inputs,
                    gold_letters[i],
                    token_masks[i],
                    processor,
                )
                
                # Compute distance for fine-tuned model
                tuned_dist = compute_attention_distance_single_sample(
                    finetuned_model,
                    single_inputs,
                    gold_letters[i],
                    token_masks[i],
                    processor,
                )
                
                if base_dist is not None and tuned_dist is not None:
                    results.append({
                        "dataset_index": idx,
                        "attn_distance_base": base_dist,
                        "attn_distance_tuned": tuned_dist,
                    })
                    
                    if not tqdm:
                        print(f"[{batch_idx * args.batch_size + i + 1}/{len(dataset)}] "
                              f"idx={idx}, base={base_dist:.6f}, tuned={tuned_dist:.6f}")
                
                sample_idx += 1  # Increment index for each sample
            
            # Free memory
            del inputs
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"Processed {len(results)} samples successfully")
    print("=" * 80)

    # Save CSV
    print(f"\nSaving results to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_index", "attn_distance_base", "attn_distance_tuned"])
        writer.writeheader()
        writer.writerows(results)

    # Create scatter plot
    print(f"Creating scatter plot at {args.output_plot}...")
    indices = [r["dataset_index"] for r in results]
    base_dists = [r["attn_distance_base"] for r in results]
    tuned_dists = [r["attn_distance_tuned"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(indices, base_dists, alpha=0.6, s=30, c='red', label='Base Model')
    ax.scatter(indices, tuned_dists, alpha=0.6, s=30, c='blue', label='Fine-tuned Model')
    
    ax.set_xlabel('Dataset Index', fontsize=12)
    ax.set_ylabel('Attention Distance (Frobenius Norm²)', fontsize=12)
    ax.set_title('Attention Distance Comparison: Base vs Fine-tuned Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output_plot}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Base Model:")
    print(f"  Mean distance: {np.mean(base_dists):.6f}")
    print(f"  Std distance:  {np.std(base_dists):.6f}")
    print(f"  Min distance:  {np.min(base_dists):.6f}")
    print(f"  Max distance:  {np.max(base_dists):.6f}")
    print(f"\nFine-tuned Model:")
    print(f"  Mean distance: {np.mean(tuned_dists):.6f}")
    print(f"  Std distance:  {np.std(tuned_dists):.6f}")
    print(f"  Min distance:  {np.min(tuned_dists):.6f}")
    print(f"  Max distance:  {np.max(tuned_dists):.6f}")
    print(f"\nImprovement:")
    improvement = np.mean(base_dists) - np.mean(tuned_dists)
    pct_improvement = (improvement / np.mean(base_dists)) * 100
    print(f"  Mean distance reduction: {improvement:.6f} ({pct_improvement:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
