#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rejection Sampling + Attention Alignment Training

Combines:
1. Self-generated chain-of-thought reasoning (rejection sampling)
2. Attention alignment loss averaged over all generated tokens
3. 4-way classification VQA loss

The model generates reasoning chains, filters by correctness, and trains
with attention alignment averaged across all generated token positions.
"""

import os
import re
import glob
import math
import time
import signal
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch


# -------------------------
# Graceful shutdown handler
# -------------------------
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        # Second interrupt - force exit
        print("\n[!] Force exit")
        sys.exit(1)
    _shutdown_requested = True
    print("\n[!] Interrupt received, finishing current step...")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from consensus_mask_dataloader import ConsensusMaskDataset

# -------------------------
# Env
# -------------------------
os.environ.setdefault("HF_HOME", "/mnt/shared/shared_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================================
# Helpers (reused from v4)
# ============================================================================

def resolve_hf_snapshot_dir(model_dir: str) -> str:
    p = Path(model_dir)
    if (p / "config.json").exists():
        return str(p)
    snaps = sorted(glob.glob(str(p / "snapshots" / "*")), key=os.path.getmtime)
    if not snaps:
        raise FileNotFoundError(f"No snapshots under: {model_dir}")
    snap = Path(snaps[-1])
    if not (snap / "config.json").exists():
        raise FileNotFoundError(f"Snapshot missing config.json: {snap}")
    return str(snap)


def _guess_repo_id(models_dir: str) -> str:
    base = Path(models_dir).name
    m = re.match(r"models--([^/]+)--(.+)$", base)
    if not m:
        raise ValueError(f"Can't infer repo id from: {models_dir}")
    return f"{m.group(1)}/{m.group(2)}"


def _hub_cache(models_dir: str) -> str:
    p = Path(models_dir)
    return str(p.parent) if p.parent.name == "hub" else str(p.parent / "hub")


def _unwrap(m):
    """Unwrap PEFT model to get base model."""
    return getattr(m, "base_model", m)


# ============================================================================
# Model loading (reused from v4)
# ============================================================================

def load_model_and_processor(
    model_path: str,
    gpu_id: int,
    attn_implementation: str = "flash_attention_2",
    use_gradient_checkpointing: bool = True,
    quantization: str = "4bit",
) -> Tuple[torch.nn.Module, AutoProcessor]:
    snapshot = resolve_hf_snapshot_dir(model_path)
    print(f"[model] Loading from: {snapshot}")

    # Processor
    try:
        processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    except (OSError, ValueError):
        repo_id = _guess_repo_id(model_path)
        processor = AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, local_files_only=True, cache_dir=_hub_cache(model_path)
        )

    # Quantization config
    load_kwargs = {}
    if quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu_id}"},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    # Prepare for training
    if quantization != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # Add LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # Gradient checkpointing
    if use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("[model] Gradient checkpointing enabled")

    return model, processor


# ============================================================================
# Cached constants - computed once at startup (reused from v4)
# ============================================================================

class TrainCache:
    """Constants computed once and reused every step."""
    
    def __init__(self, model, processor, device):
        self.device = device
        
        # Letter token IDs (A, B, C, D)
        # IMPORTANT: Use " A" (with space prefix) because in "The answer is A." 
        # the letter follows a space and gets tokenized with the space prefix.
        # Using standalone "A" would find the wrong position (in the choices section).
        tokenizer = processor.tokenizer
        self.letters = ["A", "B", "C", "D"]
        # Get token IDs for letters WITH space prefix (as they appear in "The answer is X.")
        self.letter_ids = []
        for L in self.letters:
            # Tokenize " A" and take the last token (in case space is separate)
            toks = tokenizer(" " + L, add_special_tokens=False)["input_ids"]
            self.letter_ids.append(toks[-1])
        
        # Find last attention module
        base = _unwrap(model)
        self.attn_mod = None
        for name, mod in base.named_modules():
            if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
                self.attn_mod = mod
        if self.attn_mod is None:
            raise RuntimeError("Could not find attention module")
        
        # Head geometry from config
        cfg = model.config.text_config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.head_dim = cfg.hidden_size // self.num_heads
        self.gqa_rep = self.num_heads // self.num_kv_heads
        
        # Vision token IDs
        self.vs_id = model.config.vision_start_token_id
        self.ve_id = model.config.vision_end_token_id
        
        # Assistant turn marker token IDs (for finding first generated token)
        # Qwen uses <|im_start|>assistant pattern
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_id = tokenizer("assistant", add_special_tokens=False)["input_ids"][0]
        
        print(f"[cache] heads={self.num_heads}, kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")
        print(f"[cache] letter_ids (with space prefix): {list(zip(self.letters, self.letter_ids))}")
        print(f"[cache] im_start_id={self.im_start_id}, im_end_id={self.im_end_id}, assistant_id={self.assistant_id}")


# ============================================================================
# Position finding utilities (reused from v4)
# ============================================================================

def find_answer_pos(input_ids, attn_mask, gold_tid):
    """Find position of gold answer token (search from end)."""
    seq_len = attn_mask.sum().item()
    ids = input_ids[:seq_len]
    # Search last 64 tokens first
    tail = ids[max(0, seq_len-64):]
    matches = (tail == gold_tid).nonzero(as_tuple=False)
    if len(matches) > 0:
        return max(0, seq_len-64) + matches[-1, 0].item()
    # Fallback: full search
    matches = (ids == gold_tid).nonzero(as_tuple=False)
    if len(matches) > 0:
        return matches[-1, 0].item()
    raise RuntimeError("Gold token not found")


def find_vision_span(input_ids, cache):
    """Find (start, end) of vision tokens."""
    vs = (input_ids == cache.vs_id).nonzero(as_tuple=False)
    ve = (input_ids == cache.ve_id).nonzero(as_tuple=False)
    if len(vs) == 0 or len(ve) == 0:
        return None
    return (vs[0, 0].item(), ve[0, 0].item())


def find_first_generated_pos(input_ids, attn_mask, cache):
    """
    Find position of first token generated by the model (first token of assistant response).
    
    For Qwen chat format, this is the first content token after <|im_start|>assistant\n
    We find the LAST occurrence of <|im_start|> followed by assistant token, then return
    the position after that (skipping the newline token as well).
    """
    seq_len = attn_mask.sum().item()
    ids = input_ids[:seq_len]
    
    # Find all positions of <|im_start|>
    im_start_positions = (ids == cache.im_start_id).nonzero(as_tuple=False)
    if len(im_start_positions) == 0:
        raise RuntimeError("Could not find <|im_start|> token")
    
    # Look for the last <|im_start|> that is followed by "assistant"
    # This identifies the assistant turn (not user/system turns)
    for idx in reversed(im_start_positions[:, 0].tolist()):
        # Check if next token is "assistant"
        if idx + 1 < seq_len and ids[idx + 1].item() == cache.assistant_id:
            # Found assistant turn. The first generated token is after:
            # <|im_start|> (idx) -> assistant (idx+1) -> \n (idx+2) -> FIRST_TOKEN (idx+3)
            first_gen_pos = idx + 3
            if first_gen_pos < seq_len:
                return first_gen_pos
            raise RuntimeError(f"First generated position {first_gen_pos} >= seq_len {seq_len}")
    
    raise RuntimeError("Could not find assistant turn marker")


# ============================================================================
# Forward pass with hidden state capture (reused from v4)
# ============================================================================

def forward_and_capture(model, inputs, cache):
    """
    Single forward pass that captures hidden states from last attention layer.
    Returns: (last_hidden_state, captured_hs_for_attn)
    
    For VLMs, we must use the full model (not the inner core) because it handles
    the multimodal inputs (images -> vision encoder -> language model).
    """
    captured = {}
    
    def hook(mod, args, kwargs):
        hs = args[0] if len(args) > 0 and torch.is_tensor(args[0]) else kwargs.get("hidden_states")
        if hs is not None:
            captured["hs"] = hs
    
    handle = cache.attn_mod.register_forward_pre_hook(hook, with_kwargs=True)
    
    try:
        # Use full model with output_hidden_states=True
        # This is required for VLMs because the core model doesn't handle image inputs
        out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        
        # Get last hidden state from the output
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            last_hs = out.hidden_states[-1]
        else:
            raise RuntimeError("Model did not return hidden_states")
    finally:
        handle.remove()
    
    return last_hs, captured.get("hs")


# ============================================================================
# VQA Loss (reused from v4)
# ============================================================================

def compute_vqa_loss(model, hidden_states, inputs, golds, cache):
    """4-way classification loss."""
    B = hidden_states.shape[0]
    W = model.get_output_embeddings().weight
    W4 = W[cache.letter_ids]  # (4, D)
    
    hs_list, ys = [], []
    for b in range(B):
        y = cache.letters.index(golds[b])
        pos = find_answer_pos(inputs["input_ids"][b], inputs["attention_mask"][b], cache.letter_ids[y])
        hs_list.append(hidden_states[b, pos - 1])
        ys.append(y)
    
    H = torch.stack(hs_list)
    logits = H @ W4.t()
    targets = torch.tensor(ys, device=logits.device)
    return F.cross_entropy(logits, targets)


# ============================================================================
# Chain-of-Thought Prompting
# ============================================================================

def format_cot_prompt(question: str, choices: List[str]) -> str:
    """
    Format prompt that encourages chain-of-thought reasoning before answering.
    
    Unlike the original format_prompt, this asks the model to think step by step
    about what it sees in the image before giving the final answer.
    """
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{chr(65+i)}. {c}")
    lines.extend([
        "",
        "Think step by step about what you see in the image.",
        "Then give your final answer as a single capital letter (A, B, C, or D)."
    ])
    return "\n".join(lines)


# ============================================================================
# Reasoning Chain Generation
# ============================================================================

def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract the final answer letter (A, B, C, or D) from generated text.
    
    Looks for patterns like:
    - "The answer is A"
    - "Answer: B"
    - "Therefore, C"
    - Just "D" at the end
    """
    text = text.strip()
    
    # Common patterns for final answer
    patterns = [
        r"(?:the\s+)?answer\s+is\s*[:\s]*([A-D])",
        r"answer\s*[:\s]+([A-D])",
        r"therefore[,\s]+([A-D])",
        r"so\s+(?:the\s+answer\s+is\s+)?([A-D])",
        r"\b([A-D])\s*$",  # Single letter at end
        r"\b([A-D])\s*[.\n]$",  # Letter followed by period/newline at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Fallback: look for any standalone A, B, C, D in last 50 chars
    tail = text[-50:] if len(text) > 50 else text
    for letter in ["A", "B", "C", "D"]:
        if letter in tail.upper():
            # Check it's somewhat standalone (not part of a word)
            idx = tail.upper().rfind(letter)
            if idx >= 0:
                before = tail[idx-1] if idx > 0 else " "
                after = tail[idx+1] if idx < len(tail) - 1 else " "
                if not before.isalpha() and not after.isalpha():
                    return letter
    
    return None


def generate_reasoning_chains(
    model, 
    processor, 
    image: Image.Image, 
    question: str, 
    choices: List[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> List[Tuple[str, Optional[str]]]:
    """
    Generate N reasoning chains for a single example.
    
    Args:
        model: The VLM model
        processor: The processor for tokenization
        image: PIL Image
        question: The question text
        choices: List of answer choices
        num_samples: Number of chains to generate
        max_new_tokens: Max tokens for generation
        temperature: Sampling temperature
        device: CUDA device
    
    Returns:
        List of (reasoning_text, predicted_letter) tuples.
        predicted_letter is None if no answer could be extracted.
    """
    # Build generation prompt (without gold answer - just user turn)
    prompt = format_cot_prompt(question, choices)
    msgs = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]
    
    # Process for generation
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
    img_in = img_in[0] if isinstance(img_in, list) and len(img_in) == 1 else img_in
    
    inputs = processor(
        text=[text], 
        images=[img_in], 
        videos=None, 
        padding=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)
    
    # Store the prompt length to extract only generated tokens later
    prompt_len = inputs["input_ids"].shape[1]
    
    # Generate with sampling
    with torch.no_grad():
        # Need to temporarily enable use_cache for generation
        original_use_cache = model.config.use_cache
        model.config.use_cache = True
        
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_samples,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        finally:
            model.config.use_cache = original_use_cache
    
    # Decode and extract answers
    results = []
    for i in range(num_samples):
        # Get only the generated part (excluding prompt)
        generated_ids = outputs[i, prompt_len:]
        generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract the answer letter
        predicted = extract_answer_letter(generated_text)
        results.append((generated_text.strip(), predicted))
    
    return results


def generate_reasoning_chains_batched(
    model, 
    processor, 
    images: List[Image.Image], 
    questions: List[str], 
    choices_list: List[List[str]],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> List[List[Tuple[str, Optional[str]]]]:
    """
    Generate N reasoning chains for multiple examples in a single batched call.
    
    This is much faster than calling generate_reasoning_chains() in a loop
    because it processes all images in parallel on the GPU.
    
    Args:
        model: The VLM model
        processor: The processor for tokenization
        images: List of PIL Images
        questions: List of question texts
        choices_list: List of answer choice lists
        num_samples: Number of chains to generate per example
        max_new_tokens: Max tokens for generation
        temperature: Sampling temperature
        device: CUDA device
    
    Returns:
        List of lists, where each inner list contains (reasoning_text, predicted_letter) 
        tuples for that image. Outer list has length len(images), inner lists have 
        length num_samples.
    """
    B = len(images)
    if B == 0:
        return []
    
    # Set left padding for correct batched generation with decoder-only models
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'
    
    try:
        # Build prompts for all images
        texts = []
        imgs_processed = []
        
        for img, question, choices in zip(images, questions, choices_list):
            prompt = format_cot_prompt(question, choices)
            msgs = [
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
            ]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            
            img_in, _ = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
            img_in = img_in[0] if isinstance(img_in, list) and len(img_in) == 1 else img_in
            imgs_processed.append(img_in)
        
        # Process all inputs together (with left padding)
        inputs = processor(
            text=texts, 
            images=imgs_processed, 
            videos=None, 
            padding=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)
        
        # With left padding, all sequences are padded to the same length
        # Generated tokens start after the full padded input length
        padded_len = inputs["input_ids"].shape[1]
        
        # Generate with sampling
        with torch.no_grad():
            original_use_cache = model.config.use_cache
            model.config.use_cache = True
            
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature if temperature > 0 else None,
                    num_return_sequences=num_samples,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            finally:
                model.config.use_cache = original_use_cache
        
        # outputs shape: (B * num_samples, seq_len)
        # Organization: [img0_s0, img0_s1, ..., img0_sN, img1_s0, img1_s1, ..., img1_sN, ...]
        
        # Decode and organize results
        all_results = []
        for b in range(B):
            image_results = []
            
            for s in range(num_samples):
                output_idx = b * num_samples + s
                # Get only the generated part (after the padded input)
                generated_ids = outputs[output_idx, padded_len:]
                generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract the answer letter
                predicted = extract_answer_letter(generated_text)
                image_results.append((generated_text.strip(), predicted))
            
            all_results.append(image_results)
        
        return all_results
    
    finally:
        # Restore original padding side
        processor.tokenizer.padding_side = original_padding_side


def select_best_chain(
    chains: List[Tuple[str, Optional[str]]], 
    gold_letter: str
) -> Tuple[str, bool]:
    """
    Select the best chain from generated options.
    
    Strategy:
    1. If any chain has the correct answer, return the first correct one
    2. If none are correct, return the first chain anyway
    
    Args:
        chains: List of (reasoning_text, predicted_letter) tuples
        gold_letter: The correct answer letter
    
    Returns:
        (selected_reasoning_text, is_correct)
    """
    # First, try to find a correct chain
    for reasoning, predicted in chains:
        if predicted == gold_letter:
            return reasoning, True
    
    # No correct chain found - use first one
    if chains:
        return chains[0][0], False
    
    # Edge case: no chains at all
    return "", False


# ============================================================================
# Data Processing with Reasoning
# ============================================================================

def build_batch_inputs_with_reasoning(
    processor, 
    images: List[Image.Image], 
    questions: List[str],
    choices_list: List[List[str]],
    reasonings: List[str], 
    golds: List[str]
) -> Tuple[Dict, List[Tuple[int, int]]]:
    """
    Build batch inputs that include generated reasoning in the assistant turn.
    
    Args:
        processor: The processor for tokenization
        images: List of PIL Images
        questions: List of question strings
        choices_list: List of answer choice lists
        reasonings: List of generated reasoning texts
        golds: List of gold answer letters
    
    Returns:
        inputs: Dict of batched inputs ready for model
        gen_positions: List of (start, end) positions for generated tokens per batch item
    """
    texts, imgs, vids = [], [], []
    
    for img, question, choices, reasoning, gold in zip(images, questions, choices_list, reasonings, golds):
        prompt = format_cot_prompt(question, choices)
        
        # Build assistant response: reasoning + answer
        # Format: "reasoning text\n\nThe answer is X."
        if reasoning:
            assistant_response = f"{reasoning}\n\nThe answer is {gold}."
        else:
            assistant_response = f"The answer is {gold}."
        
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
        ]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        img_in, vid_in = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
        imgs.append(img_in[0] if isinstance(img_in, list) and len(img_in) == 1 else img_in)
        vids.append(vid_in[0] if isinstance(vid_in, list) and len(vid_in) == 1 else vid_in)
    
    vids = [v for v in vids if v is not None and (not isinstance(v, list) or len(v) > 0)]
    inputs = processor(text=texts, images=imgs, videos=vids if vids else None, padding=True, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    
    return inputs


def find_generated_span(input_ids, attn_mask, cache) -> Tuple[int, int]:
    """
    Find the span (start, end) of generated tokens in the sequence.
    
    Generated tokens are everything after <|im_start|>assistant\n up to (but not including) <|im_end|>.
    
    Returns:
        (start_pos, end_pos) where generated tokens are at positions [start_pos, end_pos)
    """
    seq_len = attn_mask.sum().item()
    ids = input_ids[:seq_len]
    
    # Find all positions of <|im_start|>
    im_start_positions = (ids == cache.im_start_id).nonzero(as_tuple=False)
    if len(im_start_positions) == 0:
        raise RuntimeError("Could not find <|im_start|> token")
    
    # Find the last <|im_start|> followed by "assistant" (the assistant turn)
    start_pos = None
    for idx in reversed(im_start_positions[:, 0].tolist()):
        if idx + 1 < seq_len and ids[idx + 1].item() == cache.assistant_id:
            # <|im_start|> (idx) -> assistant (idx+1) -> \n (idx+2) -> FIRST_TOKEN (idx+3)
            start_pos = idx + 3
            break
    
    if start_pos is None:
        raise RuntimeError("Could not find assistant turn marker")
    
    # Find <|im_end|> token ID and locate the end of assistant response
    im_end_id = cache.im_start_id  # Usually <|im_end|> has different ID, but let's search for it
    # Actually, we need to get im_end_id. Let's search for the pattern after start_pos
    
    # For Qwen, look for <|im_end|> which ends the assistant turn
    # We'll search from start_pos to end of sequence
    remaining = ids[start_pos:]
    
    # Try to find <|im_end|> - typically it's a special token
    # If we can't find it, use the actual sequence length
    end_pos = seq_len
    
    # Look for common end markers or padding
    # The safest is to find where padding starts or sequence ends
    # Usually: [... response tokens ...] <|im_end|> [padding]
    # We want end_pos to be the position of the last content token + 1
    
    # Count backwards from the end to find where content ends
    for i in range(seq_len - 1, start_pos - 1, -1):
        tok_id = ids[i].item()
        # Skip padding tokens and <|im_end|> special token
        if tok_id not in [0, cache.im_end_id]:
            end_pos = i + 1  # Position after last non-special token
            break
    
    # Ensure we have at least some generated tokens
    if end_pos <= start_pos:
        end_pos = start_pos + 1
    
    return (start_pos, end_pos)


def make_collate_fn_simple(processor):
    """
    Simple collate function for the dataloader.
    Unlike v4, we don't build inputs here - we just collect the raw data.
    Input building happens in train_step after reasoning generation.
    """
    def collate(batch):
        images = [b["image"].convert("RGB") for b in batch]
        questions = [b["question"] for b in batch]
        choices = [b["answer_choices"] for b in batch]
        golds = [b["gold_letter"] for b in batch]
        masks = [b["token_mask"] for b in batch]
        return {
            "images": images,
            "questions": questions,
            "choices": choices,
            "golds": golds,
            "masks": masks,
        }
    return collate


# ============================================================================
# Attention Alignment Loss
# ============================================================================

def compute_attn_loss(
    hidden_states, 
    inputs, 
    golds,
    masks, 
    cache, 
    captured_hs, 
    mode: str = "average",
    eps: float = 1e-8
):
    """
    Attention alignment loss with configurable token position(s).
    
    Args:
        hidden_states: Last hidden states from model output
        inputs: Batched model inputs
        golds: List of gold answer letters (needed for "last" mode)
        masks: List of attention target masks (one per batch item)
        cache: TrainCache with model constants
        captured_hs: Hidden states captured at last attention layer
        mode: Which token(s) to compute attention alignment for:
            - "first": Only the first generated token (after image+prompt)
            - "last": Only the answer token position
            - "average": Average over all generated token positions
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss tensor (averaged over batch and positions)
    """
    B = hidden_states.shape[0]
    device = hidden_states.device
    batch_losses = []
    
    for b in range(B):
        # Find the span of vision tokens
        span = find_vision_span(inputs["input_ids"][b], cache)
        if span is None:
            continue
        
        vs, ve = span
        I = ve - vs - 1  # Number of vision tokens
        if I <= 0:
            continue
        
        # Target distribution from mask
        m = masks[b].to(device=device, dtype=torch.bfloat16).flatten()
        if m.numel() != I or m.sum() <= 0:
            continue
        t = (m + eps) / (m.sum() + eps * I)
        
        # Determine which position(s) to compute attention loss at
        if mode == "last":
            # Use the answer token position
            y = cache.letters.index(golds[b])
            try:
                ans_pos = find_answer_pos(
                    inputs["input_ids"][b], 
                    inputs["attention_mask"][b], 
                    cache.letter_ids[y]
                )
                positions = [ans_pos]
            except RuntimeError:
                continue
        else:
            # Need generated span for "first" and "average"
            try:
                gen_start, gen_end = find_generated_span(
                    inputs["input_ids"][b], 
                    inputs["attention_mask"][b], 
                    cache
                )
            except RuntimeError:
                continue
            
            if gen_end <= gen_start:
                continue
            
            if mode == "first":
                positions = [gen_start]
            else:  # "average"
                positions = list(range(gen_start, gen_end))
        
        if not positions:
            continue
        
        # Pre-compute K projections for vision tokens (same for all positions)
        k_raw = cache.attn_mod.k_proj(captured_hs[b:b+1, vs+1:ve, :])
        k = k_raw.view(1, I, cache.num_kv_heads, cache.head_dim).transpose(1, 2)
        if cache.gqa_rep != 1:
            k = k.repeat_interleave(cache.gqa_rep, dim=1)
        
        # Compute attention alignment loss at each position
        position_losses = []
        for pos in positions:
            # Q projection for this position
            q_raw = cache.attn_mod.q_proj(captured_hs[b:b+1, pos:pos+1, :])
            q = q_raw.view(1, 1, cache.num_heads, cache.head_dim).transpose(1, 2)
            
            # Attention scores
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(cache.head_dim)
            p = torch.softmax(scores, dim=-1).squeeze(2).mean(dim=1).squeeze(0)
            p = (p + eps) / (p.sum() + eps * I)
            
            # Squared error loss
            loss_at_pos = ((p - t) ** 2).sum()
            position_losses.append(loss_at_pos)
        
        if position_losses:
            # Average over selected positions (single for first/last, all for average)
            batch_losses.append(torch.stack(position_losses).mean())
    
    if not batch_losses:
        return torch.zeros((), device=device, requires_grad=True)
    
    return torch.stack(batch_losses).mean()


# ============================================================================
# Training Step
# ============================================================================

def train_step(
    model, 
    processor,
    batch, 
    device, 
    cache, 
    use_attn_loss: bool,
    lambda_attn: float,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    txt_token_for_attn: str = "average",
    batch_generations: bool = False,
    debug_timing: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Single training step with rejection sampling.
    
    Phase 1: Generate reasoning chains for each example (no grad)
    Phase 2: Build training inputs with selected reasoning
    Phase 3: Forward pass and loss computation (with grad)
    
    Args:
        txt_token_for_attn: Which token(s) to use for attention alignment:
            - "first": First generated token
            - "last": Answer token
            - "average": All generated tokens (default)
        batch_generations: If True, generate all reasoning chains in one batched call (faster)
    
    Returns:
        (total_loss, vqa_loss, attn_loss, correct_ratio, avg_reasoning_len)
    """
    images = batch["images"]
    questions = batch["questions"]
    choices = batch["choices"]
    golds = batch["golds"]
    masks = batch["masks"]
    B = len(images)
    
    if debug_timing:
        torch.cuda.synchronize(device)
        t0 = time.time()
    
    # ========================================================================
    # Phase 1: Generate reasoning chains (no grad)
    # ========================================================================
    selected_reasonings = []
    num_correct = 0
    total_reasoning_tokens = 0
    
    model.eval()  # Switch to eval mode for generation
    
    if batch_generations:
        # Batched generation: process all images in one call
        all_chains = generate_reasoning_chains_batched(
            model=model,
            processor=processor,
            images=images,
            questions=questions,
            choices_list=choices,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        
        for i in range(B):
            chains = all_chains[i]
            reasoning, is_correct = select_best_chain(chains, golds[i])
            selected_reasonings.append(reasoning)
            
            if is_correct:
                num_correct += 1
            total_reasoning_tokens += len(processor.tokenizer.encode(reasoning, add_special_tokens=False))
    else:
        # Sequential generation: process one image at a time
        for i in range(B):
            chains = generate_reasoning_chains(
                model=model,
                processor=processor,
                image=images[i],
                question=questions[i],
                choices=choices[i],
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            
            reasoning, is_correct = select_best_chain(chains, golds[i])
            selected_reasonings.append(reasoning)
            
            if is_correct:
                num_correct += 1
            total_reasoning_tokens += len(processor.tokenizer.encode(reasoning, add_special_tokens=False))
    
    correct_ratio = num_correct / B
    avg_reasoning_len = total_reasoning_tokens / B
    
    model.train()  # Switch back to train mode
    
    if debug_timing:
        torch.cuda.synchronize(device)
        t1 = time.time()
        print(f"    [timing] generation: {t1-t0:.3f}s, correct_ratio={correct_ratio:.2f}")
    
    # ========================================================================
    # Phase 2: Build training inputs with selected reasoning
    # ========================================================================
    inputs = build_batch_inputs_with_reasoning(
        processor=processor,
        images=images,
        questions=questions,
        choices_list=choices,
        reasonings=selected_reasonings,
        golds=golds,
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    if debug_timing:
        torch.cuda.synchronize(device)
        t2 = time.time()
        seq_len = inputs["input_ids"].shape[1]
        print(f"    [timing] build_inputs: {t2-t1:.3f}s, seq_len={seq_len}")
    
    # ========================================================================
    # Phase 3: Forward pass and loss computation (with grad)
    # ========================================================================
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        last_hs, captured_hs = forward_and_capture(model, inputs, cache)
        
        if debug_timing:
            torch.cuda.synchronize(device)
            t3 = time.time()
            print(f"    [timing] forward: {t3-t2:.3f}s")
        
        # VQA loss (4-way classification at position before answer token)
        vqa_loss = compute_vqa_loss(model, last_hs, inputs, golds, cache)
        
        if debug_timing:
            torch.cuda.synchronize(device)
            t4 = time.time()
            print(f"    [timing] vqa_loss: {t4-t3:.3f}s")
        
        # Attention alignment loss
        if use_attn_loss and captured_hs is not None:
            attn_loss = compute_attn_loss(
                last_hs, inputs, golds, masks, cache, captured_hs,
                mode=txt_token_for_attn
            )
            total_loss = vqa_loss + lambda_attn * attn_loss
            
            if debug_timing:
                torch.cuda.synchronize(device)
                t5 = time.time()
                print(f"    [timing] attn_loss: {t5-t4:.3f}s")
        else:
            attn_loss = torch.tensor(0.0, device=device)
            total_loss = vqa_loss
    
    return total_loss, vqa_loss, attn_loss, correct_ratio, avg_reasoning_len


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rejection Sampling + Attention Alignment Training"
    )
    
    # Model and data
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_path", type=str, 
                        default="/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct")
    parser.add_argument("--consensus_dir", type=str, required=True,
                        help="Directory with consensus mask data")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (note: effective throughput is lower due to generation)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lambda_attn", type=float, default=25.0,
                        help="Weight for attention alignment loss")
    
    # Generation parameters (rejection sampling)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of reasoning chains to generate per example")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens for reasoning generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    parser.add_argument("--batch_generations", action="store_true",
                        help="Batch generation across all images in the batch (faster but uses more VRAM)")
    
    # Model configuration
    parser.add_argument("--no_ckpt", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--quantization", type=str, default="4bit", 
                        choices=["none", "4bit", "8bit"])
    
    # Loss function
    parser.add_argument("--loss_func", type=str, default="attention_alignment", 
                        choices=["vqa_only", "attention_alignment"],
                        help="Which loss to use")
    parser.add_argument("--txt_token_for_attn", type=str, default="average",
                        choices=["first", "last", "average"],
                        help="Which token(s) to compute attention alignment on: "
                             "first=first generated token, last=answer token, "
                             "average=all generated tokens")
    
    # Output and logging
    parser.add_argument("--save_dir", type=str, default="./lora_rs",
                        help="Directory to save trained model")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--tqdm", action="store_true",
                        help="Use tqdm progress bar")
    
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    
    # ========================================================================
    # Load model
    # ========================================================================
    model, processor = load_model_and_processor(
        args.model_path, 
        args.gpu,
        use_gradient_checkpointing=(not args.no_ckpt),
        quantization=args.quantization,
    )
    
    # Build cache
    cache = TrainCache(model, processor, device)
    
    # ========================================================================
    # Dataset
    # ========================================================================
    dataset = ConsensusMaskDataset(
        consensus_dir=args.consensus_dir,
        model=model, 
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
        shuffle=True,
        num_workers=0, 
        pin_memory=True, 
        collate_fn=make_collate_fn_simple(processor),
    )
    
    # ========================================================================
    # Optimizer
    # ========================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # ========================================================================
    # Training loop
    # ========================================================================
    model.train()
    use_attn = (args.loss_func == "attention_alignment")
    total_steps = args.n_epochs * len(loader)
    
    pbar = tqdm(total=total_steps, desc="train") if args.tqdm and tqdm else None
    
    print(f"[train] epochs={args.n_epochs}, steps/epoch={len(loader)}, batch_size={args.batch_size}")
    print(f"[train] loss={args.loss_func}, lambda_attn={args.lambda_attn}, txt_token_for_attn={args.txt_token_for_attn}")
    print(f"[train] num_samples={args.num_samples}, max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, batch_gen={args.batch_generations}")
    
    step = 0
    interrupted = False
    debug_timing = True  # Enable detailed timing for first few steps
    
    # Running averages for logging
    running_correct_ratio = 0.0
    running_reasoning_len = 0.0
    
    try:
        for epoch in range(args.n_epochs):
            t_batch_start = time.time()
            
            for batch in loader:
                t_batch_loaded = time.time()
                
                # Check for interrupt
                if _shutdown_requested:
                    print("[!] Stopping training...")
                    interrupted = True
                    break
                
                if debug_timing and step < 3:
                    print(f"  [timing] dataloader: {t_batch_loaded - t_batch_start:.3f}s")
                
                t0 = time.time()
                
                loss, vqa_loss, attn_loss, correct_ratio, avg_reasoning_len = train_step(
                    model=model,
                    processor=processor,
                    batch=batch,
                    device=device,
                    cache=cache,
                    use_attn_loss=use_attn,
                    lambda_attn=args.lambda_attn,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    txt_token_for_attn=args.txt_token_for_attn,
                    batch_generations=args.batch_generations,
                    debug_timing=(debug_timing and step < 3),
                )
                
                t1 = time.time()
                
                # Update running averages
                alpha = 0.1
                running_correct_ratio = alpha * correct_ratio + (1 - alpha) * running_correct_ratio
                running_reasoning_len = alpha * avg_reasoning_len + (1 - alpha) * running_reasoning_len
                
                (loss / args.grad_accum).backward()
                
                t2 = time.time()
                
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                # Logging
                if pbar:
                    mem = torch.cuda.memory_allocated(device) / 1e9
                    pbar.set_postfix(
                        loss=f"{loss.item():.3f}", 
                        vqa=f"{vqa_loss.item():.3f}", 
                        attn=f"{attn_loss.item():.3f}",
                        cr=f"{running_correct_ratio:.2f}",
                        mem=f"{mem:.1f}G"
                    )
                    pbar.update(1)
                elif step % args.log_every == 0:
                    mem = torch.cuda.memory_allocated(device) / 1e9
                    print(f"[{epoch+1}/{args.n_epochs}] step={step} loss={loss.item():.4f} "
                          f"vqa={vqa_loss.item():.4f} attn={attn_loss.item():.4f} "
                          f"cr={running_correct_ratio:.2f} rlen={running_reasoning_len:.0f} "
                          f"fwd={t1-t0:.2f}s bwd={t2-t1:.2f}s mem={mem:.1f}G")
                
                step += 1
                t_batch_start = time.time()
            
            if interrupted:
                break
    
    except Exception as e:
        print(f"[!] Error during training: {e}")
        import traceback
        traceback.print_exc()
        interrupted = True
    
    finally:
        # Cleanup
        if pbar:
            pbar.close()
        
        # Synchronize CUDA before exit
        torch.cuda.synchronize(device)
    
    # ========================================================================
    # Save (even if interrupted, save partial progress)
    # ========================================================================
    if step > 0:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        processor.save_pretrained(args.save_dir)
        if interrupted:
            print(f"[saved] Partial checkpoint ({step} steps) saved to {args.save_dir}")
        else:
            print(f"[done] Saved to {args.save_dir}")
    else:
        print("[!] No steps completed, nothing saved")


if __name__ == "__main__":
    main()
