#!/usr/bin/env python3
"""
Debug script to understand tokenization and VQA loss computation.
Tests the first 5 samples to see token IDs and loss values.
"""

import os
import sys
import glob
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from consensus_mask_dataloader import ConsensusMaskDataset

os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def resolve_hf_snapshot_dir(model_dir: str) -> str:
    p = Path(model_dir)
    if (p / "config.json").exists():
        return str(p)
    snaps = sorted(glob.glob(str(p / "snapshots" / "*")), key=os.path.getmtime)
    if not snaps:
        raise FileNotFoundError(f"No snapshots under: {model_dir}")
    snap = Path(snaps[-1])
    return str(snap)


def main():
    gpu = 0
    model_path = "/mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct"
    consensus_dir = "/mnt/arc/mjojic/causal-patch-selection/segment_patches/consensus_search/consensus_patches"
    
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    
    # Load model
    snapshot = resolve_hf_snapshot_dir(model_path)
    print(f"Loading model from: {snapshot}")
    
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer
    
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu}"},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.eval()
    
    print("\n" + "="*80)
    print("TOKENIZATION ANALYSIS")
    print("="*80)
    
    # Test tokenization of letters
    letters = ["A", "B", "C", "D"]
    print("\n1. Standalone letter tokenization:")
    standalone_ids = []
    for L in letters:
        toks = tokenizer(L, add_special_tokens=False)["input_ids"]
        standalone_ids.append(toks[0])
        decoded = tokenizer.decode(toks)
        print(f"   '{L}' -> IDs: {toks}, decoded: '{decoded}'")
    
    print("\n2. Space-prefixed letter tokenization:")
    space_prefixed_ids = []
    for L in letters:
        toks = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
        space_prefixed_ids.append(toks[-1])  # Take last token
        decoded = tokenizer.decode(toks)
        print(f"   ' {L}' -> IDs: {toks}, decoded: '{decoded}', using last: {toks[-1]}")
    
    print("\n3. 'The answer is X.' tokenization:")
    for L in letters:
        text = f"The answer is {L}."
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded_toks = [tokenizer.decode([t]) for t in toks]
        print(f"   '{text}' -> {list(zip(toks, decoded_toks))}")
    
    # Load dataset
    print("\n" + "="*80)
    print("TESTING ON FIRST 5 SAMPLES")
    print("="*80)
    
    dataset = ConsensusMaskDataset(
        consensus_dir=consensus_dir,
        model=model, 
        processor=processor,
        precompute_token_masks=True, 
        use_precomputed_masks=True,
        cache_images=False, 
        cache_masks=False, 
        max_side=1024,
    )
    
    # Get output embeddings
    W = model.get_output_embeddings().weight
    print(f"\nOutput embeddings shape: {W.shape}, dtype: {W.dtype}")
    
    for sample_idx in range(min(5, len(dataset))):
        print(f"\n{'='*80}")
        print(f"SAMPLE {sample_idx}")
        print("="*80)
        
        sample = dataset[sample_idx]
        image = sample["image"].convert("RGB")
        question = sample["question"]
        choices = sample["answer_choices"]
        gold = sample["gold_letter"]
        
        print(f"Question: {question[:100]}...")
        print(f"Choices: {choices}")
        print(f"Gold answer: {gold}")
        
        # Build prompt (like format_cot_prompt)
        lines = [question.strip(), ""]
        for i, c in enumerate(choices):
            lines.append(f"{chr(65+i)}. {c}")
        lines.extend([
            "",
            "Think step by step about what you see in the image.",
            "Then give your final answer as a single capital letter (A, B, C, or D)."
        ])
        prompt = "\n".join(lines)
        
        # Fake reasoning for testing
        reasoning = "Looking at the image, I can see the relevant details."
        assistant_response = f"{reasoning}\n\nThe answer is {gold}."
        
        # Build messages
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
        ]
        
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        img_in, vid_in = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
        img_in = img_in[0] if isinstance(img_in, list) and len(img_in) == 1 else img_in
        
        inputs = processor(text=[text], images=[img_in], videos=None, padding=True, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"][0]
        attn_mask = inputs["attention_mask"][0]
        seq_len = attn_mask.sum().item()
        
        print(f"\nSequence length: {seq_len}")
        
        # Show last 30 tokens
        print(f"\nLast 30 tokens:")
        for i in range(max(0, seq_len-30), seq_len):
            tok_id = input_ids[i].item()
            tok_str = tokenizer.decode([tok_id])
            print(f"   pos {i}: ID={tok_id}, '{tok_str}'")
        
        # Find where standalone and space-prefixed IDs appear
        gold_idx = letters.index(gold)
        standalone_id = standalone_ids[gold_idx]
        space_id = space_prefixed_ids[gold_idx]
        
        print(f"\nSearching for gold='{gold}':")
        print(f"   Standalone ID: {standalone_id}")
        print(f"   Space-prefixed ID: {space_id}")
        
        # Find all occurrences
        ids = input_ids[:seq_len]
        standalone_matches = (ids == standalone_id).nonzero(as_tuple=False)
        space_matches = (ids == space_id).nonzero(as_tuple=False)
        
        print(f"   Standalone matches at positions: {standalone_matches.flatten().tolist()}")
        print(f"   Space-prefixed matches at positions: {space_matches.flatten().tolist()}")
        
        # Which one would find_answer_pos use?
        def find_answer_pos_debug(input_ids, attn_mask, gold_tid, name):
            seq_len = attn_mask.sum().item()
            ids = input_ids[:seq_len]
            tail = ids[max(0, seq_len-64):]
            matches = (tail == gold_tid).nonzero(as_tuple=False)
            if len(matches) > 0:
                pos = max(0, seq_len-64) + matches[-1, 0].item()
                print(f"   {name}: Found in tail at pos {pos}")
                return pos
            matches = (ids == gold_tid).nonzero(as_tuple=False)
            if len(matches) > 0:
                pos = matches[-1, 0].item()
                print(f"   {name}: Found in full search at pos {pos}")
                return pos
            print(f"   {name}: NOT FOUND!")
            return None
        
        pos_standalone = find_answer_pos_debug(input_ids, attn_mask, standalone_id, "Standalone")
        pos_space = find_answer_pos_debug(input_ids, attn_mask, space_id, "Space-prefixed")
        
        # Compute VQA loss for both
        print(f"\nVQA Loss computation:")
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
                hidden_states = out.hidden_states[-1]
        
        for pos, name, letter_ids in [
            (pos_standalone, "Standalone IDs", standalone_ids),
            (pos_space, "Space-prefixed IDs", space_prefixed_ids),
        ]:
            if pos is None:
                print(f"   {name}: Cannot compute (position not found)")
                continue
            
            W4 = W[letter_ids].to(hidden_states.dtype)
            h = hidden_states[0, pos - 1]  # Hidden state before answer token
            logits = h @ W4.t()
            probs = F.softmax(logits, dim=-1)
            
            target = letters.index(gold)
            loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([target], device=device))
            
            print(f"\n   {name}:")
            print(f"      Position used: {pos} (hidden state at {pos-1})")
            print(f"      Token at pos {pos}: ID={input_ids[pos].item()}, '{tokenizer.decode([input_ids[pos].item()])}'")
            print(f"      Token at pos {pos-1}: ID={input_ids[pos-1].item()}, '{tokenizer.decode([input_ids[pos-1].item()])}'")
            print(f"      Logits: {logits.tolist()}")
            print(f"      Probs:  {probs.tolist()}")
            print(f"      Target: {target} ({gold})")
            print(f"      Loss:   {loss.item():.4f}")


if __name__ == "__main__":
    main()
