#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention Alignment Training - Full Fine-tuning (No LoRA)

This script performs full fine-tuning of the Qwen VL model without LoRA adapters.
It has the same training logic as attention_alignment_v4.py but trains all parameters.

Focus on essential optimizations only:
1. Single forward pass (major memory savings)
2. Cached constants (letter IDs, attention module)
3. Simple, readable code (no over-optimization)
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
# Helpers
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


# ============================================================================
# Model loading - Full fine-tuning (no LoRA)
# ============================================================================

def load_model_and_processor(
    model_path: str,
    gpu_id: int,
    attn_implementation: str = "flash_attention_2",
    use_gradient_checkpointing: bool = True,
    freeze_vision: bool = False,
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

    # Load model in bfloat16 (full precision, no quantization)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu_id}"},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    # Enable gradients for all parameters
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Optionally freeze vision encoder
    if freeze_vision:
        if hasattr(model, "visual"):
            for param in model.visual.parameters():
                param.requires_grad = False
            print("[model] Vision encoder frozen")
        elif hasattr(model, "vision_tower"):
            for param in model.vision_tower.parameters():
                param.requires_grad = False
            print("[model] Vision tower frozen")
        else:
            print("[model] Warning: Could not find vision encoder to freeze")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

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
# Cached constants - computed once at startup
# ============================================================================

class TrainCache:
    """Constants computed once and reused every step."""
    
    def __init__(self, model, processor, device):
        self.device = device
        
        # Letter token IDs (A, B, C, D)
        tokenizer = processor.tokenizer
        self.letters = ["A", "B", "C", "D"]
        self.letter_ids = [tokenizer(L, add_special_tokens=False)["input_ids"][0] for L in self.letters]
        
        # Find last attention module (direct access, no PEFT wrapper)
        self.attn_mod = None
        for name, mod in model.named_modules():
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
        self.assistant_id = tokenizer("assistant", add_special_tokens=False)["input_ids"][0]
        
        print(f"[cache] heads={self.num_heads}, kv_heads={self.num_kv_heads}, head_dim={self.head_dim}")
        print(f"[cache] im_start_id={self.im_start_id}, assistant_id={self.assistant_id}")


# ============================================================================
# Data processing
# ============================================================================

def format_prompt(question: str, choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{chr(65+i)}. {c}")
    lines.extend(["", "Answer with a single capital letter (A, B, C, D, ...)."])
    return "\n".join(lines)


def build_batch_inputs(processor, images, prompts, golds):
    texts, imgs, vids = [], [], []
    for img, prompt, gold in zip(images, prompts, golds):
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": gold}]},
        ]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        img_in, vid_in = process_vision_info(msgs, image_patch_size=processor.image_processor.patch_size)
        imgs.append(img_in[0] if isinstance(img_in, list) and len(img_in) == 1 else img_in)
        vids.append(vid_in[0] if isinstance(vid_in, list) and len(vid_in) == 1 else vid_in)
    
    vids = [v for v in vids if v is not None and (not isinstance(v, list) or len(v) > 0)]
    inputs = processor(text=texts, images=imgs, videos=vids if vids else None, padding=True, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    return inputs


def make_collate_fn(processor):
    def collate(batch):
        images = [b["image"].convert("RGB") for b in batch]
        prompts = [format_prompt(b["question"], b["answer_choices"]) for b in batch]
        golds = [b["gold_letter"] for b in batch]
        masks = [b["token_mask"] for b in batch]
        inputs = build_batch_inputs(processor, images, prompts, golds)
        return {"inputs": inputs, "golds": golds, "masks": masks}
    return collate


# ============================================================================
# Loss computation
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


def compute_attn_loss(hidden_states, inputs, golds, masks, cache, captured_hs, txt_token_for_attn="last", eps=1e-8):
    """
    Attention alignment loss.
    
    Args:
        txt_token_for_attn: Which text token to extract attention from.
            - "last": Use the gold answer token position (A/B/C/D) - original behavior
            - "first": Use the first generated token (first token of assistant response)
    """
    B = hidden_states.shape[0]
    device = hidden_states.device
    losses = []
    
    for b in range(B):
        # Find the text token position based on setting
        if txt_token_for_attn == "last":
            # Use gold answer token position (original behavior)
            y = cache.letters.index(golds[b])
            txt_pos = find_answer_pos(inputs["input_ids"][b], inputs["attention_mask"][b], cache.letter_ids[y])
        else:  # "first"
            # Use first generated token position
            txt_pos = find_first_generated_pos(inputs["input_ids"][b], inputs["attention_mask"][b], cache)
        
        span = find_vision_span(inputs["input_ids"][b], cache)
        if span is None:
            continue
        
        vs, ve = span
        I = ve - vs - 1
        if I <= 0:
            continue
        
        # Target distribution from mask
        m = masks[b].to(device=device, dtype=torch.bfloat16).flatten()
        if m.numel() != I or m.sum() <= 0:
            continue
        t = (m + eps) / (m.sum() + eps * I)
        
        # Q/K projections from captured hidden states
        q_raw = cache.attn_mod.q_proj(captured_hs[b:b+1, txt_pos:txt_pos+1, :])
        k_raw = cache.attn_mod.k_proj(captured_hs[b:b+1, vs+1:ve, :])
        
        # Reshape
        q = q_raw.view(1, 1, cache.num_heads, cache.head_dim).transpose(1, 2)
        k = k_raw.view(1, I, cache.num_kv_heads, cache.head_dim).transpose(1, 2)
        
        if cache.gqa_rep != 1:
            k = k.repeat_interleave(cache.gqa_rep, dim=1)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(cache.head_dim)
        p = torch.softmax(scores, dim=-1).squeeze(2).mean(dim=1).squeeze(0)
        p = (p + eps) / (p.sum() + eps * I)
        
        losses.append(((p - t) ** 2).sum())
    
    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ============================================================================
# Forward pass with hidden state capture
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
# Training step
# ============================================================================

def train_step(model, batch, device, cache, use_attn_loss, lambda_attn, txt_token_for_attn="last", debug_timing=False):
    """Single training step."""
    
    if debug_timing:
        torch.cuda.synchronize(device)
        t0 = time.time()
    
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["inputs"].items()}
    golds = batch["golds"]
    masks = batch["masks"]
    
    if debug_timing:
        torch.cuda.synchronize(device)
        t1 = time.time()
        # Print sequence info
        seq_len = inputs["input_ids"].shape[1]
        print(f"    [timing] to_device: {t1-t0:.3f}s, seq_len={seq_len}, batch={inputs['input_ids'].shape[0]}")
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if debug_timing:
            torch.cuda.synchronize(device)
            t2 = time.time()
        
        last_hs, captured_hs = forward_and_capture(model, inputs, cache)
        
        if debug_timing:
            torch.cuda.synchronize(device)
            t3 = time.time()
            print(f"    [timing] model_forward: {t3-t2:.3f}s")
        
        vqa_loss = compute_vqa_loss(model, last_hs, inputs, golds, cache)
        
        if debug_timing:
            torch.cuda.synchronize(device)
            t4 = time.time()
            print(f"    [timing] vqa_loss: {t4-t3:.3f}s")
        
        if use_attn_loss and captured_hs is not None:
            attn_loss = compute_attn_loss(last_hs, inputs, golds, masks, cache, captured_hs, txt_token_for_attn)
            total_loss = vqa_loss + lambda_attn * attn_loss
            
            if debug_timing:
                torch.cuda.synchronize(device)
                t5 = time.time()
                print(f"    [timing] attn_loss: {t5-t4:.3f}s")
        else:
            attn_loss = torch.tensor(0.0, device=device)
            total_loss = vqa_loss
    
    return total_loss, vqa_loss, attn_loss


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="/mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct")
    parser.add_argument("--consensus_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (lower than LoRA since we're training all params)")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lambda_attn", type=float, default=25.0)
    parser.add_argument("--save_dir", type=str, default="./checkpoint_full")
    parser.add_argument("--no_ckpt", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--freeze_vision", action="store_true", help="Freeze vision encoder (only train language model)")
    parser.add_argument("--loss_func", type=str, default="attention_alignment", choices=["vqa_only", "attention_alignment"])
    parser.add_argument("--txt_token_for_attn", type=str, default="last", choices=["first", "last"],
                        help="Which text token to extract attention from: 'last' (answer token) or 'first' (first generated token)")
    parser.add_argument("--tqdm", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Load model (full fine-tuning, no LoRA)
    model, processor = load_model_and_processor(
        args.model_path, args.gpu,
        use_gradient_checkpointing=(not args.no_ckpt),
        freeze_vision=args.freeze_vision,
    )

    # Build cache
    cache = TrainCache(model, processor, device)

    # Dataset
    dataset = ConsensusMaskDataset(
        consensus_dir=args.consensus_dir,
        model=model, processor=processor,
        precompute_token_masks=True, use_precomputed_masks=True,
        cache_images=False, cache_masks=False, max_side=1024,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=make_collate_fn(processor),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training
    model.train()
    use_attn = (args.loss_func == "attention_alignment")
    total_steps = args.n_epochs * len(loader)
    
    pbar = tqdm(total=total_steps, desc="train") if args.tqdm and tqdm else None
    
    print(f"[train] epochs={args.n_epochs}, steps/epoch={len(loader)}, batch_size={args.batch_size}")
    print(f"[train] loss={args.loss_func}, lambda_attn={args.lambda_attn}, txt_token_for_attn={args.txt_token_for_attn}")
    print(f"[train] Full fine-tuning mode (no LoRA), freeze_vision={args.freeze_vision}")
    
    step = 0
    interrupted = False
    debug_timing = True  # Enable detailed timing for first few steps
    
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
                
                if debug_timing and step < 5:
                    print(f"  [timing] dataloader/collate: {t_batch_loaded - t_batch_start:.3f}s")
                
                t0 = time.time()
                
                loss, vqa_loss, attn_loss = train_step(
                    model, batch, device, cache, use_attn, args.lambda_attn,
                    txt_token_for_attn=args.txt_token_for_attn,
                    debug_timing=(debug_timing and step < 5)
                )
                
                t1 = time.time()
                
                (loss / args.grad_accum).backward()
                
                t2 = time.time()
                
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                # Logging - print every batch
                mem = torch.cuda.memory_allocated(device) / 1e9
                print(f"[{epoch+1}/{args.n_epochs}] step={step} loss={loss.item():.4f} "
                      f"vqa={vqa_loss.item():.4f} attn={attn_loss.item():.4f} "
                      f"fwd={t1-t0:.2f}s bwd={t2-t1:.2f}s mem={mem:.1f}G")
                
                if pbar:
                    pbar.set_postfix(loss=f"{loss.item():.3f}", vqa=f"{vqa_loss.item():.3f}", 
                                    attn=f"{attn_loss.item():.3f}", mem=f"{mem:.1f}G")
                    pbar.update(1)
                
                step += 1
                t_batch_start = time.time()  # Reset for next iteration
            
            if interrupted:
                break
    
    except Exception as e:
        print(f"[!] Error during training: {e}")
        interrupted = True
    
    finally:
        # Cleanup
        if pbar:
            pbar.close()
        
        # Synchronize CUDA before exit
        torch.cuda.synchronize(device)
    
    # Save (even if interrupted, save partial progress)
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
