import math
import torch
import argparse
import os
import re
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import numpy as np
import seaborn as sns  # kept to match original (not strictly required)
from PIL import Image

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from consensus_mask_dataloader import ConsensusMaskDataset
from attention_alignment import (
    resolve_hf_snapshot_dir,
    _guess_repo_id_from_models_dir,
    _shared_hf_hub_cache,
    format_mc_prompt,
)

# Default to your Qwen3-VL-8B-Instruct location
model_path = '/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct'

def load_model(model_id, gpu=0):
    # Resolve model path to actual snapshot directory
    snapshot_path = resolve_hf_snapshot_dir(model_id)
    print(f"[model] Resolved snapshot: {snapshot_path}")
    
    # Load processor with fallback
    try:
        processor = AutoProcessor.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side='left',
            use_fast=True
        )
    except (OSError, ValueError):
        repo_id = _guess_repo_id_from_models_dir(model_id)
        hub_cache = _shared_hf_hub_cache(model_id)
        print(f"[processor] fallback repo_id={repo_id} cache_dir={hub_cache}")
        processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=hub_cache,
            padding_side='left',
            use_fast=True
        )
    
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",      # IMPORTANT for output_attentions
        device_map={ "": f"cuda:{gpu}" },
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    return model, processor

def load_finetuned_model(model_id, lora_dir, gpu=0):
    """Load base model and then apply LoRA adapters"""
    base_model, processor = load_model(model_id, gpu=gpu)
    
    if PeftModel is None:
        raise RuntimeError("peft is not available but lora_dir was provided.")
    
    print(f"[finetuned model] Loading adapters from {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    return model, processor

def calculate_plt_size(attention_layer_num):
    cols = math.ceil(math.sqrt(attention_layer_num))
    rows = math.ceil(attention_layer_num / cols)
    return rows, cols

def _get_vision_token_ids(model, processor):
    """
    Match their style (convert_tokens_to_ids), but fall back to model.config ids
    in case tokenizer doesn't expose the literal tokens.
    """
    tok = processor.tokenizer
    try:
        vs = tok.convert_tokens_to_ids('<|vision_start|>')
        ve = tok.convert_tokens_to_ids('<|vision_end|>')
        if vs is None or ve is None or vs < 0 or ve < 0:
            raise ValueError("convert_tokens_to_ids returned invalid ids")
        return int(vs), int(ve)
    except Exception:
        vs = getattr(model.config, "vision_start_token_id", None)
        ve = getattr(model.config, "vision_end_token_id", None)
        if vs is None or ve is None:
            raise RuntimeError("Could not resolve vision start/end token ids.")
        return int(vs), int(ve)

def _infer_hw_from_len(n: int):
    """
    Simple fallback if output_shape doesn't match attention length.
    Prefer near-square.
    """
    if n <= 0:
        return (1, 1)
    r = int(math.isqrt(n))
    for h in range(r, 0, -1):
        if n % h == 0:
            w = n // h
            return (h, w)
    return (1, n)

def _create_attention_overlay(att_reshaped, image_np, img_w, img_h):
    """Create normalized attention overlay resized to match image dimensions"""
    from PIL import Image as PILImage
    
    # Normalize attention to [0, 1]
    att_min, att_max = att_reshaped.min(), att_reshaped.max()
    if att_max - att_min > 1e-8:
        att_normalized = (att_reshaped - att_min) / (att_max - att_min)
    else:
        att_normalized = np.zeros_like(att_reshaped)
    
    # Resize attention map to match image dimensions
    att_pil = PILImage.fromarray((att_normalized * 255).astype(np.uint8), mode='L')
    att_resized = att_pil.resize((img_w, img_h), resample=PILImage.BILINEAR)
    att_resized_np = np.array(att_resized).astype(np.float32) / 255.0
    
    return att_resized_np

def get_response(base_model, tuned_model, processor, image, question, gold_letter, answer_choices, final_answer_tokens=1024, out_path='heatmap.png'):
    """Generate attention heatmaps for both base and tuned models side-by-side
    
    Args:
        base_model: Base model without fine-tuning
        tuned_model: Fine-tuned model (can be None to only visualize base model)
        image: PIL Image object
        gold_letter: The correct answer letter (A/B/C/D)
        answer_choices: List of answer choice strings
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(base_model.device)

    input_ids = inputs['input_ids'][0].tolist()
    vision_start_token_id, vision_end_token_id = _get_vision_token_ids(base_model, processor)

    # Same indexing behavior as their script
    try:
        pos = input_ids.index(vision_start_token_id) + 1
        pos_end = input_ids.index(vision_end_token_id)
    except ValueError:
        raise RuntimeError("Could not find <|vision_start|>/<|vision_end|> in input_ids. Prompt/template may differ.")

    image_indices = list(range(pos, pos_end))  # kept for parity (not used below)

    # Same grid logic as their script: image_grid_thw[1:]/2
    # (Qwen VL commonly merges 2x2 patches before feeding LLM image tokens)
    try:
        image_inputs_aux = processor.image_processor(images=image_inputs)
    except TypeError:
        # some processors require return_tensors
        image_inputs_aux = processor.image_processor(images=image_inputs, return_tensors="pt")

    grid_thw = image_inputs_aux.get("image_grid_thw", None)
    if grid_thw is None:
        raise RuntimeError("processor.image_processor did not return image_grid_thw; cannot reshape heatmap grid.")

    if torch.is_tensor(grid_thw):
        grid_thw_np = grid_thw.detach().cpu().numpy()
    else:
        grid_thw_np = np.array(grid_thw)

    output_shape = grid_thw_np.squeeze(0)[1:] / 2
    output_shape = output_shape.astype(int)
    out_h, out_w = int(output_shape[0]), int(output_shape[1])

    # Generate with base model
    print("[generate] Running base model...")
    with torch.no_grad():
        base_gen_output = base_model.generate(
            **inputs,
            max_new_tokens=final_answer_tokens,
            do_sample=True,
            output_attentions=True,
            return_dict_in_generate=True
        )
        base_gen_ids = base_gen_output.sequences
        base_trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, base_gen_ids)]
        base_output_text = processor.batch_decode(base_trimmed_ids, skip_special_tokens=True)[0].strip()
        base_attentions = base_gen_output.attentions
    
    # Generate with tuned model if provided
    if tuned_model is not None:
        print("[generate] Running fine-tuned model...")
        with torch.no_grad():
            tuned_gen_output = tuned_model.generate(
                **inputs,
                max_new_tokens=final_answer_tokens,
                do_sample=True,
                output_attentions=True,
                return_dict_in_generate=True
            )
            tuned_gen_ids = tuned_gen_output.sequences
            tuned_trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, tuned_gen_ids)]
            tuned_output_text = processor.batch_decode(tuned_trimmed_ids, skip_special_tokens=True)[0].strip()
            tuned_attentions = tuned_gen_output.attentions
    else:
        tuned_attentions = None
        tuned_output_text = None

    num_layers = len(base_attentions[0])
    base_num_tokens = len(base_attentions)
    
    # Convert image to numpy array for display
    image_np = np.array(image)
    img_h, img_w = image_np.shape[:2]
    
    # Setup figure: if we have tuned model, show side-by-side (2 columns per layer)
    # Otherwise just show base model
    if tuned_attentions is not None:
        rows, cols = calculate_plt_size(num_layers)
        # Double the columns for side-by-side comparison
        fig = plt.figure(figsize=(21.6, 18))  # Increased height for text
        gs = fig.add_gridspec(rows, cols * 2, hspace=0.3, wspace=0.1, top=0.88, bottom=0.02)
        
        # Add main title
        fig.suptitle('Attention Heatmap Comparison: Base (Left) vs Fine-tuned (Right)', 
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        rows, cols = calculate_plt_size(num_layers)
        fig = plt.figure(figsize=(10.8, 18))  # Increased height for text
        gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.1, top=0.88, bottom=0.02)
        fig.suptitle('Base Model Attention Heatmaps', fontsize=16, fontweight='bold', y=0.98)
    
    # Add question and answer information at the top
    # Extract the base question from the formatted prompt (before answer choices)
    question_text = question.split('\n\nAnswer Choices:')[0] if 'Answer Choices:' in question else question
    question_text = question_text.replace('Question: ', '')
    
    # Format answer choices
    choices_text = "Answer Choices:\n"
    for i, choice in enumerate(answer_choices):
        letter = chr(65 + i)  # A, B, C, D
        marker = " ← GOLD" if letter == gold_letter else ""
        choices_text += f"  {letter}. {choice}{marker}\n"
    
    # Add text to figure
    info_text = f"Question: {question_text}\n\n{choices_text}"
    fig.text(0.5, 0.94, info_text, ha='center', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace', wrap=True)
    
    tuned_num_tokens = len(tuned_attentions) if tuned_attentions is not None else 0
    
    for layer_idx in range(num_layers):
        row = layer_idx // cols
        col = layer_idx % cols
        
        # Process base model attention for this layer
        base_layer_att = []
        for t in range(base_num_tokens):
            att_t = base_attentions[t][layer_idx][0, :, -1, pos:pos_end].mean(dim=0)
            base_layer_att.append(att_t)
        
        base_att = torch.stack(base_layer_att).mean(dim=0)
        base_att = base_att.to(torch.float32).detach().cpu().numpy()
        
        # Reshape base attention
        expected = out_h * out_w
        curr_out_h, curr_out_w = out_h, out_w
        if base_att.size != expected:
            if base_att.size == out_w * out_h:
                pass
            else:
                ih, iw = _infer_hw_from_len(base_att.size)
                curr_out_h, curr_out_w = ih, iw
        
        base_att_reshaped = base_att.reshape((curr_out_h, curr_out_w))
        base_att_overlay = _create_attention_overlay(base_att_reshaped, image_np, img_w, img_h)
        
        # Create base model subplot
        if tuned_attentions is not None:
            ax_base = fig.add_subplot(gs[row, col * 2])
        else:
            ax_base = fig.add_subplot(gs[row, col])
        
        ax_base.imshow(image_np)
        ax_base.imshow(base_att_overlay, cmap="jet", alpha=0.5, interpolation="bilinear")
        ax_base.set_title(f"Layer {layer_idx+1} - Base", fontsize=9, pad=3)
        ax_base.axis("off")
        
        # Process tuned model if available
        if tuned_attentions is not None:
            tuned_layer_att = []
            for t in range(tuned_num_tokens):
                att_t = tuned_attentions[t][layer_idx][0, :, -1, pos:pos_end].mean(dim=0)
                tuned_layer_att.append(att_t)
            
            tuned_att = torch.stack(tuned_layer_att).mean(dim=0)
            tuned_att = tuned_att.to(torch.float32).detach().cpu().numpy()
            
            # Reshape tuned attention
            if tuned_att.size != expected:
                if tuned_att.size == out_w * out_h:
                    pass
                else:
                    ih, iw = _infer_hw_from_len(tuned_att.size)
                    curr_out_h, curr_out_w = ih, iw
            
            tuned_att_reshaped = tuned_att.reshape((curr_out_h, curr_out_w))
            tuned_att_overlay = _create_attention_overlay(tuned_att_reshaped, image_np, img_w, img_h)
            
            # Create tuned model subplot
            ax_tuned = fig.add_subplot(gs[row, col * 2 + 1])
            ax_tuned.imshow(image_np)
            ax_tuned.imshow(tuned_att_overlay, cmap="jet", alpha=0.5, interpolation="bilinear")
            ax_tuned.set_title(f"Layer {layer_idx+1} - Tuned", fontsize=9, pad=3)
            ax_tuned.axis("off")
    
    # Fill remaining subplots if grid not completely filled
    total_subplots = rows * cols * (2 if tuned_attentions is not None else 1)
    needed_subplots = num_layers * (2 if tuned_attentions is not None else 1)
    if tuned_attentions is not None:
        for idx in range(needed_subplots, total_subplots):
            row = idx // (cols * 2)
            col = idx % (cols * 2)
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
    else:
        for idx in range(needed_subplots, total_subplots):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 80)
    print("Base Model Generated Answer:")
    print(base_output_text)
    if tuned_output_text is not None:
        print("\n" + "=" * 80)
        print("Fine-tuned Model Generated Answer:")
        print(tuned_output_text)
    print("=" * 80)
    
    return base_output_text, tuned_output_text

def process_single_sample(base_model, tuned_model, processor, image, question, gold_letter, answer_choices, out_path='heatmap.png'):
    """Generate heatmaps for base and optionally tuned model
    
    Args:
        base_model: Base model without fine-tuning
        tuned_model: Fine-tuned model (can be None)
        image: PIL Image object
        gold_letter: The correct answer letter
        answer_choices: List of answer choice strings
    """
    base_answer, tuned_answer = get_response(base_model, tuned_model, processor, image, question, gold_letter, answer_choices, out_path=out_path)
    return base_answer, tuned_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--consensus_dir', type=str, required=True,
                       help='Directory containing consensus patch data')
    parser.add_argument('--dataset_indices', type=int, nargs='+', required=True,
                       help='One or more dataset indices to visualize (space-separated)')
    parser.add_argument('--model_id', type=str, default=model_path)
    parser.add_argument('--lora_dir', type=str, default=None,
                       help='Optional LoRA adapter directory for fine-tuned model comparison')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final_answer_tokens', type=int, default=1024)
    parser.add_argument('--out_dir', type=str, default='.',
                       help='Output directory for heatmap images (default: current directory)')
    args = parser.parse_args()

    # Load base model
    print("=" * 80)
    print("Loading BASE model...")
    print("=" * 80)
    base_model, processor = load_model(args.model_id, gpu=args.gpu)
    
    # Load fine-tuned model if specified
    tuned_model = None
    if args.lora_dir:
        print("\n" + "=" * 80)
        print("Loading FINE-TUNED model...")
        print("=" * 80)
        tuned_model, _ = load_finetuned_model(args.model_id, args.lora_dir, gpu=args.gpu)
    
    # Load dataset
    print(f"\n[dataset] Loading from {args.consensus_dir}")
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
    print(f"[dataset] Total samples: {len(dataset)}")
    print(f"[dataset] Processing {len(args.dataset_indices)} sample(s): {args.dataset_indices}")
    
    # Process each index
    for idx_num, dataset_index in enumerate(args.dataset_indices, 1):
        print(f"\n{'='*80}")
        print(f"Processing sample {idx_num}/{len(args.dataset_indices)}: index {dataset_index}")
        print(f"{'='*80}")
        
        # Get sample at specified index
        if dataset_index < 0 or dataset_index >= len(dataset):
            print(f"[WARNING] dataset_index {dataset_index} out of range [0, {len(dataset)-1}], skipping...")
            continue
        
        sample = dataset[dataset_index]
        image = sample["image"].convert("RGB")
        
        # Format the question as multiple choice
        question = format_mc_prompt(sample["question"], sample["answer_choices"])
        gold_letter = sample["gold_letter"]
        
        print(f"\n[sample {dataset_index}]")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Choices: {sample['answer_choices']}")
        print(f"  Gold answer: {gold_letter}")
        print(f"  Image size: {image.size}")
        
        # Generate output path for this index
        out_path = os.path.join(args.out_dir, f'heatmap_sample_{dataset_index}.png')
        
        process_single_sample(base_model, tuned_model, processor, image, question, 
                            gold_letter, sample["answer_choices"], out_path=out_path)
        print(f"\n[save] {out_path}")
    
    print(f"\n{'='*80}")
    print(f"Completed processing {len(args.dataset_indices)} sample(s)")
    print(f"{'='*80}")