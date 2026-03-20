#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for VQA loss token extraction logic.

Verifies that the answer token position finding logic correctly identifies
the gold answer token (A/B/C/D) in the tokenized sequence.
"""

import os
import sys
import argparse

# Environment setup
os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Import from train_attn
from train_attn import (
    resolve_hf_snapshot_dir,
    format_prompt,
    build_batch_inputs,
    find_answer_pos,
)
from consensus_mask_dataloader import ConsensusMaskDataset


def main():
    parser = argparse.ArgumentParser(description="Test VQA loss token extraction")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--model_path", type=str, 
                        default="/mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct")
    parser.add_argument("--consensus_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load processor only (we don't need the full model for this test)
    print(f"[test] Loading processor from {args.model_path}")
    snapshot = resolve_hf_snapshot_dir(args.model_path)
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer
    
    # Get letter token IDs
    letters = ["A", "B", "C", "D"]
    letter_ids = [tokenizer(L, add_special_tokens=False)["input_ids"][0] for L in letters]
    print(f"[test] Letter token IDs: {dict(zip(letters, letter_ids))}")
    
    # Load dataset
    print(f"[test] Loading dataset from {args.consensus_dir}")
    
    # We need a minimal model config for the dataset - load just config
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
    )
    
    dataset = ConsensusMaskDataset(
        consensus_dir=args.consensus_dir,
        model=model, processor=processor,
        precompute_token_masks=True, use_precomputed_masks=True,
        cache_images=False, cache_masks=False, max_side=1024,
    )
    
    print(f"[test] Dataset size: {len(dataset)}")
    print(f"[test] Testing first {args.n_samples} samples\n")
    print("=" * 80)
    
    all_match = True
    
    for i in range(min(args.n_samples, len(dataset))):
        sample = dataset[i]
        
        # Build inputs for single sample
        image = sample["image"].convert("RGB")
        prompt = format_prompt(sample["question"], sample["answer_choices"])
        gold = sample["gold_letter"]
        
        inputs = build_batch_inputs(processor, [image], [prompt], [gold])
        
        # Get the expected token ID for the gold letter
        gold_idx = letters.index(gold)
        gold_tid = letter_ids[gold_idx]
        
        # Find the answer position
        try:
            pos = find_answer_pos(inputs["input_ids"][0], inputs["attention_mask"][0], gold_tid)
        except RuntimeError as e:
            print(f"Sample {i}: ERROR - {e}")
            all_match = False
            continue
        
        # Extract the token at that position
        extracted_tid = inputs["input_ids"][0, pos].item()
        extracted_token = tokenizer.decode([extracted_tid])
        
        # Check if they match
        match = (extracted_tid == gold_tid)
        status = "✓ MATCH" if match else "✗ MISMATCH"
        if not match:
            all_match = False
        
        # Also show the token before (which is used for prediction)
        prev_tid = inputs["input_ids"][0, pos - 1].item()
        prev_token = tokenizer.decode([prev_tid])
        
        # Print results
        print(f"Sample {i}:")
        print(f"  Question: {sample['question'][:60]}...")
        print(f"  Expected gold letter: '{gold}' (token_id={gold_tid})")
        print(f"  Found position: {pos}")
        print(f"  Token at pos {pos}: '{extracted_token}' (token_id={extracted_tid})")
        print(f"  Token at pos {pos-1} (used for prediction): '{prev_token}' (token_id={prev_tid})")
        print(f"  Status: {status}")
        
        # Show context around the answer
        seq_len = inputs["attention_mask"][0].sum().item()
        start = max(0, pos - 5)
        end = min(seq_len, pos + 3)
        context_ids = inputs["input_ids"][0, start:end].tolist()
        context_tokens = [tokenizer.decode([tid]) for tid in context_ids]
        
        # Mark the answer position
        answer_idx_in_context = pos - start
        context_display = []
        for j, tok in enumerate(context_tokens):
            if j == answer_idx_in_context:
                context_display.append(f"[{tok}]")  # Highlight answer
            else:
                context_display.append(tok)
        
        print(f"  Context: {''.join(context_display)}")
        print()
    
    print("=" * 80)
    if all_match:
        print("All samples PASSED - answer token extraction is working correctly!")
    else:
        print("Some samples FAILED - check the mismatches above.")
    
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
