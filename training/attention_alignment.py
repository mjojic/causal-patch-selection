
import os
import re
import glob
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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


# ============================================================================
# HF snapshot helpers
# ============================================================================

def resolve_hf_snapshot_dir(model_dir: str) -> str:
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


def _guess_repo_id_from_models_dir(models_dir: str) -> str:
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


# ============================================================================
# Model loading
# ============================================================================

def load_model_and_processor(
    model_path: str,
    gpu_id: int,
    attn_implementation: str = "flash_attention_2",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_gradient_checkpointing: bool = True,
    quantization: str = "4bit",          # "none" | "4bit" | "8bit"
    bnb_compute_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    snapshot_path = resolve_hf_snapshot_dir(model_path)
    print(f"[model] Resolved snapshot: {snapshot_path}")

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

    quantization = (quantization or "none").lower()
    if quantization not in {"none", "4bit", "8bit"}:
        raise ValueError(f"quantization must be one of: none|4bit|8bit (got {quantization})")

    load_kwargs = {}
    if quantization != "none":
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig import failed. Install transformers + bitsandbytes.")
        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )
            print("[model] Using 4-bit NF4 quantization (QLoRA-style).")
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("[model] Using 8-bit quantization.")
    else:
        load_kwargs["dtype"] = torch.bfloat16

    print(f"[model] Loading on GPU {gpu_id} with attn_implementation='{attn_implementation}', quant={quantization}")
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": f"cuda:{gpu_id}"},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    if quantization != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    if use_lora:
        print(f"[LoRA] Adding adapters r={lora_r}, alpha={lora_alpha}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.config.use_cache = False

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    return model, processor


# ============================================================================
# Prompt + batching
# ============================================================================

def format_mc_prompt(question: str, answer_choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for i, c in enumerate(answer_choices):
        letter = chr(65 + i)
        lines.append(f"{letter}. {c}")
    lines.append("")
    lines.append("Answer with a single capital letter (A, B, C, D, ...).")
    return "\n".join(lines)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_inputs_for_batch(
    processor: AutoProcessor,
    images: List[Image.Image],
    prompts: List[str],
    gold_letters: List[str],
) -> Dict[str, torch.Tensor]:
    texts = []
    img_list = []
    vid_list = []

    for img, prompt, gold in zip(images, prompts, gold_letters):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": gold}]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(
            messages, image_patch_size=processor.image_processor.patch_size
        )

        img_list.append(image_inputs[0] if isinstance(image_inputs, list) and len(image_inputs) == 1 else image_inputs)
        vid_list.append(video_inputs[0] if isinstance(video_inputs, list) and len(video_inputs) == 1 else video_inputs)
        texts.append(text)

    vid_list_filtered = [v for v in vid_list if v is not None and (not isinstance(v, list) or len(v) > 0)]

    inputs = processor(
        text=texts,
        images=img_list,
        videos=vid_list_filtered if vid_list_filtered else None,
        padding=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs


def collate_fn_factory(processor: AutoProcessor):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [b["image"].convert("RGB") for b in batch]
        prompts = [format_mc_prompt(b["question"], b["answer_choices"]) for b in batch]
        gold_letters = [b["gold_letter"] for b in batch]
        token_masks = [b["token_mask"] for b in batch]  # variable (h,w)

        inputs = build_inputs_for_batch(processor, images, prompts, gold_letters)
        return {"inputs": inputs, "gold_letters": gold_letters, "token_masks": token_masks}
    return collate


# ============================================================================
# Vision span + head meta
# ============================================================================

def find_vision_spans(input_ids_1d: torch.Tensor, model) -> List[Tuple[int, int]]:
    vs_id = model.config.vision_start_token_id
    ve_id = model.config.vision_end_token_id
    vs_pos = (input_ids_1d == vs_id).nonzero(as_tuple=False).flatten().tolist()
    ve_pos = (input_ids_1d == ve_id).nonzero(as_tuple=False).flatten().tolist()

    spans = []
    if not vs_pos or not ve_pos:
        return spans
    j = 0
    for s in vs_pos:
        while j < len(ve_pos) and ve_pos[j] < s:
            j += 1
        if j < len(ve_pos):
            spans.append((s, ve_pos[j]))
            j += 1
    return spans


def _unwrap_peft(m):
    return getattr(m, "base_model", m)


def find_last_attention_module(model) -> Tuple[str, torch.nn.Module]:
    cand = []
    for name, mod in model.named_modules():
        if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
            cand.append((name, mod))
    if not cand:
        raise RuntimeError("Could not find any module with q_proj/k_proj.")
    return cand[-1]


def _pick_text_config(model, attn_mod):
    mcfg = getattr(model, "config", None)
    if mcfg is not None and getattr(mcfg, "text_config", None) is not None:
        return mcfg.text_config
    if getattr(attn_mod, "config", None) is not None:
        return attn_mod.config
    if mcfg is not None:
        return mcfg
    return None


def get_head_meta(attn_mod, model, q_raw: torch.Tensor, k_raw: torch.Tensor) -> Tuple[int, int, int]:
    cfg = _pick_text_config(model, attn_mod)
    if cfg is None:
        raise RuntimeError("Could not locate config to infer attention head meta.")

    num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_kv_heads", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "dim", None)

    head_dim = getattr(cfg, "head_dim", None) or getattr(cfg, "attention_head_dim", None)
    if head_dim is None and (hidden_size is not None) and (num_heads is not None):
        head_dim = int(hidden_size) // int(num_heads)

    if num_heads is None:
        if head_dim is None:
            raise RuntimeError("Cannot infer num_heads (missing num_heads and head_dim/hidden_size).")
        if q_raw.shape[-1] % head_dim != 0:
            raise RuntimeError(f"q_proj width {q_raw.shape[-1]} not divisible by head_dim {head_dim}")
        num_heads = q_raw.shape[-1] // head_dim

    if num_kv_heads is None:
        if head_dim is None:
            raise RuntimeError("Cannot infer num_kv_heads (missing head_dim/hidden_size).")
        if k_raw.shape[-1] % head_dim != 0:
            raise RuntimeError(f"k_proj width {k_raw.shape[-1]} not divisible by head_dim {head_dim}")
        num_kv_heads = k_raw.shape[-1] // head_dim

    return int(num_heads), int(num_kv_heads), int(head_dim)


# ============================================================================
# Core forward WITHOUT vocab logits
# ============================================================================

def forward_core_model_for_hidden_states(model, inputs: Dict[str, torch.Tensor]):
    base = _unwrap_peft(model)
    core = getattr(base, "model", None)
    if core is None:
        out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            last_h = out.hidden_states[-1]
            class HiddenStateWrapper:
                def __init__(self, h):
                    self.last_hidden_state = h
            return HiddenStateWrapper(last_h), "full_model"
        return out, "full_model"

    out = core(**inputs, return_dict=True, use_cache=False)
    if not hasattr(out, "last_hidden_state"):
        out_full = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        if hasattr(out_full, "hidden_states") and out_full.hidden_states is not None:
            last_h = out_full.hidden_states[-1]
            class HiddenStateWrapper:
                def __init__(self, h):
                    self.last_hidden_state = h
            return HiddenStateWrapper(last_h), "full_model"
        return out_full, "full_model"
    return out, "core:model"


# ============================================================================
# Losses
# ============================================================================

def get_letter_token_ids(tokenizer) -> Tuple[List[str], List[int]]:
    letters = ["A", "B", "C", "D"]
    ids = []
    for L in letters:
        enc = tokenizer(L, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise RuntimeError(f"Tokenizer does not map '{L}' to 1 token: ids={enc}. Adjust loss logic.")
        ids.append(enc[0])
    return letters, ids


def find_answer_token_pos(input_ids_1d: torch.Tensor, attn_mask_1d: torch.Tensor, gold_token_id: int) -> int:
    S = int(attn_mask_1d.sum().item())
    tail_start = max(0, S - 64)
    tail = input_ids_1d[tail_start:S]
    hits = (tail == gold_token_id).nonzero(as_tuple=False).flatten().tolist()
    if not hits:
        hits2 = (input_ids_1d[:S] == gold_token_id).nonzero(as_tuple=False).flatten().tolist()
        if not hits2:
            raise RuntimeError("Could not find gold answer token id in input_ids.")
        return hits2[-1]
    return tail_start + hits[-1]


def vqa_4way_loss_from_hidden(
    model,
    processor,
    last_hidden_state: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    gold_letters: List[str],
) -> torch.Tensor:
    tokenizer = processor.tokenizer
    letters, letter_ids = get_letter_token_ids(tokenizer)

    W = model.get_output_embeddings().weight  # (V,D)
    W4 = W[torch.tensor(letter_ids, device=last_hidden_state.device)]  # (4,D)

    B, S, D = last_hidden_state.shape
    attn_mask = inputs["attention_mask"]

    ys = []
    hs = []
    for b in range(B):
        gold = gold_letters[b]
        if gold not in letters:
            raise RuntimeError(f"Gold letter '{gold}' not in {letters}. Expand letters if needed.")
        y = letters.index(gold)
        gold_tid = letter_ids[y]

        ans_pos = find_answer_token_pos(inputs["input_ids"][b], attn_mask[b], gold_tid)
        if ans_pos <= 0:
            raise RuntimeError("Answer token at pos 0; cannot use ans_pos-1.")

        hs.append(last_hidden_state[b, ans_pos - 1, :])
        ys.append(y)

    H = torch.stack(hs, dim=0)
    logits4 = H @ W4.t()
    y = torch.tensor(ys, device=logits4.device)

    return F.cross_entropy(logits4.float(), y)


def attention_loss_frobenius_fast_qk_lastlayer(
    model,
    inputs: Dict[str, torch.Tensor],
    gold_letters: List[str],
    token_masks_hw: List[torch.Tensor],
    processor,
    eps: float = 1e-8,
) -> torch.Tensor:
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
            if hs is None:
                raise RuntimeError(f"Could not locate hidden_states in hook. kwargs={list(kwargs.keys())}")
        captured["hs"] = hs
        return None

    h = attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        _core_out, _mode = forward_core_model_for_hidden_states(model, inputs)
    finally:
        h.remove()

    hs_all = captured.get("hs", None)
    if hs_all is None:
        raise RuntimeError("Hook did not capture hidden_states.")

    device = hs_all.device
    B, S, D = hs_all.shape

    letters, letter_ids = get_letter_token_ids(processor.tokenizer)

    losses = []
    for b in range(B):
        gold = gold_letters[b]
        if gold not in letters:
            continue
        y = letters.index(gold)
        gold_tid = letter_ids[y]
        ans_pos = find_answer_token_pos(inputs["input_ids"][b], inputs["attention_mask"][b], gold_tid)

        spans = find_vision_spans(inputs["input_ids"][b], model)
        if not spans:
            continue
        vs, ve = spans[0]
        img_pos = torch.arange(vs + 1, ve, device=device)
        I = img_pos.numel()
        if I <= 0:
            continue

        m_hw = token_masks_hw[b].to(device=device, dtype=torch.bool)
        m = m_hw.flatten().to(dtype=torch.float32)
        if m.numel() != I:
            raise RuntimeError(f"[b={b}] token mask elems={m.numel()} but vision span implies I={I}.")
        if m.sum() <= 0:
            continue

        q_raw = attn_mod.q_proj(hs_all[b:b+1, ans_pos:ans_pos+1, :])
        k_raw = attn_mod.k_proj(hs_all[b:b+1, img_pos, :])

        num_heads, num_kv_heads, head_dim = get_head_meta(attn_mod, base, q_raw, k_raw)

        q = q_raw.view(1, 1, num_heads, head_dim).transpose(1, 2)
        k = k_raw.view(1, I, num_kv_heads, head_dim).transpose(1, 2)

        rep = num_heads // num_kv_heads
        if rep * num_kv_heads != num_heads:
            raise RuntimeError("GQA mismatch: num_heads not divisible by num_kv_heads.")
        if rep != 1:
            k = k.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        p = torch.softmax(scores, dim=-1).squeeze(2).mean(dim=1).squeeze(0)
        p = (p + eps) / (p.sum() + eps * I)

        t = (m + eps) / (m.sum() + eps * I)
        losses.append(((p - t) ** 2).sum())

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# ============================================================================
# Train
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--model_path", type=str,
                        default="/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct")
    parser.add_argument("--consensus_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps (overrides n_epochs if set)")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lambda_attn", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./attn_align_lora")
    parser.add_argument("--no_ckpt", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--quantization", type=str, default="4bit",
                        choices=["none", "4bit", "8bit"])
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm progress bar")
    parser.add_argument("--loss_func", type=str, default="attention_alignment",
                        choices=["vqa_only", "attention_alignment"],
                        help="Loss function: vqa_only (only VQA loss) or attention_alignment (VQA + attention alignment loss)")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    model, processor = load_model_and_processor(
        args.model_path,
        gpu_id=args.gpu,
        attn_implementation=args.attn_implementation,
        use_gradient_checkpointing=(not args.no_ckpt),
        quantization=args.quantization,
    )
    device = next(model.parameters()).device
    print(f"[device] {device}")

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
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_factory(processor),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    opt.zero_grad(set_to_none=True)

    # Determine total steps
    if args.max_steps is not None:
        total_steps = args.max_steps
        use_epochs = False
        print(f"[train] Running for {total_steps} steps (max_steps mode)")
    else:
        total_steps = args.n_epochs * len(loader)
        use_epochs = True
        print(f"[train] Running for {args.n_epochs} epochs ({len(loader)} batches/epoch, {total_steps} total steps)")

    use_bar = bool(args.tqdm and tqdm is not None)
    pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True) if use_bar else None

    global_step = 0
    print(f"[train] loss_func={args.loss_func}")
    if args.loss_func == "vqa_only":
        print("[train] Using VQA loss only (no attention alignment)")
    else:
        print(f"[train] Using VQA + attention alignment loss (lambda_attn={args.lambda_attn})")
    print("[train] starting...")

    if use_epochs:
        # Epoch-based training
        for epoch in range(args.n_epochs):
            if not use_bar:
                print(f"\n=== Epoch {epoch + 1}/{args.n_epochs} ===")
            
            for batch_idx, batch in enumerate(loader):
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

                inputs = _to_device(batch["inputs"], device)
                gold_letters = batch["gold_letters"]
                token_masks = batch["token_masks"]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    core_out, mode = forward_core_model_for_hidden_states(model, inputs)
                    if not hasattr(core_out, "last_hidden_state"):
                        raise RuntimeError(f"Core forward did not return last_hidden_state (mode={mode}).")
                    last_h = core_out.last_hidden_state

                    vqa_loss = vqa_4way_loss_from_hidden(model, processor, last_h, inputs, gold_letters)
                    
                    if args.loss_func == "attention_alignment":
                        attn_loss = attention_loss_frobenius_fast_qk_lastlayer(
                            model=model,
                            inputs=inputs,
                            gold_letters=gold_letters,
                            token_masks_hw=token_masks,
                            processor=processor,
                        )
                        loss = vqa_loss + args.lambda_attn * attn_loss
                    else:
                        attn_loss = torch.zeros((), device=device)
                        loss = vqa_loss

                (loss / args.grad_accum).backward()

                if (global_step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # progress updates
                if pbar is not None:
                    cur = torch.cuda.memory_allocated(device) / (1024**3)
                    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                    postfix = {
                        "epoch": epoch + 1,
                        "loss": f"{loss.item():.3g}",
                        "vqa": f"{vqa_loss.item():.3g}",
                        "mem": f"{cur:.2f}G",
                    }
                    if args.loss_func == "attention_alignment":
                        postfix["attn"] = f"{attn_loss.item():.3g}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                if (not use_bar) and (global_step % args.log_every == 0):
                    cur = torch.cuda.memory_allocated(device) / (1024**3)
                    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                    if args.loss_func == "attention_alignment":
                        print(
                            f"[epoch {epoch + 1}/{args.n_epochs}] [step {global_step:05d}] "
                            f"loss={loss.item():.4g} vqa={vqa_loss.item():.4g} attn={attn_loss.item():.4g} "
                            f"mem_cur={cur:.2f}GiB mem_peak={peak:.2f}GiB"
                        )
                    else:
                        print(
                            f"[epoch {epoch + 1}/{args.n_epochs}] [step {global_step:05d}] "
                            f"loss={loss.item():.4g} vqa={vqa_loss.item():.4g} "
                            f"mem_cur={cur:.2f}GiB mem_peak={peak:.2f}GiB"
                        )

                global_step += 1
            
            if args.max_steps is not None and global_step >= args.max_steps:
                break
    else:
        # Step-based training (legacy, with infinite shuffle)
        step = 0
        epoch = 0
        while step < total_steps:
            for batch in loader:
                if step >= total_steps:
                    break

                inputs = _to_device(batch["inputs"], device)
                gold_letters = batch["gold_letters"]
                token_masks = batch["token_masks"]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    core_out, mode = forward_core_model_for_hidden_states(model, inputs)
                    if not hasattr(core_out, "last_hidden_state"):
                        raise RuntimeError(f"Core forward did not return last_hidden_state (mode={mode}).")
                    last_h = core_out.last_hidden_state

                    vqa_loss = vqa_4way_loss_from_hidden(model, processor, last_h, inputs, gold_letters)
                    
                    if args.loss_func == "attention_alignment":
                        attn_loss = attention_loss_frobenius_fast_qk_lastlayer(
                            model=model,
                            inputs=inputs,
                            gold_letters=gold_letters,
                            token_masks_hw=token_masks,
                            processor=processor,
                        )
                        loss = vqa_loss + args.lambda_attn * attn_loss
                    else:
                        attn_loss = torch.zeros((), device=device)
                        loss = vqa_loss

                (loss / args.grad_accum).backward()

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # progress updates
                if pbar is not None:
                    cur = torch.cuda.memory_allocated(device) / (1024**3)
                    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                    postfix = {
                        "loss": f"{loss.item():.3g}",
                        "vqa": f"{vqa_loss.item():.3g}",
                        "mem": f"{cur:.2f}G",
                    }
                    if args.loss_func == "attention_alignment":
                        postfix["attn"] = f"{attn_loss.item():.3g}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                if (not use_bar) and (step % args.log_every == 0):
                    cur = torch.cuda.memory_allocated(device) / (1024**3)
                    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                    if args.loss_func == "attention_alignment":
                        print(
                            f"[step {step:05d}] loss={loss.item():.4g} "
                            f"vqa={vqa_loss.item():.4g} attn={attn_loss.item():.4g} "
                            f"mem_cur={cur:.2f}GiB mem_peak={peak:.2f}GiB"
                        )
                    else:
                        print(
                            f"[step {step:05d}] loss={loss.item():.4g} "
                            f"vqa={vqa_loss.item():.4g} "
                            f"mem_cur={cur:.2f}GiB mem_peak={peak:.2f}GiB"
                        )

                step += 1
            epoch += 1

    if pbar is not None:
        pbar.close()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[save] saving to {args.save_dir}")
    model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)
    print("[done]")


if __name__ == "__main__":
    main()