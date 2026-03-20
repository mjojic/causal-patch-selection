#!/usr/bin/env python3
"""
Inspect processor output to understand vision token mapping.
This mimics how search.py prepares inputs for vLLM.
"""
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

print("="*80)
print("PROCESSOR OUTPUT INSPECTION")
print("="*80)

# Use the model path from your search.py - this is the hub cache dir
model_dir = "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8"
image_path = "/mnt/arc/mjojic/causal-patch-selection/segment_patches/found_patches/seg_comparison_1.png"

print(f"\n[1] Loading processor...")
# Load processor the same way as search.py (with fallback)
repo_id = "Qwen/Qwen3-VL-32B-Instruct-FP8"
hub_cache = "/mnt/shared/shared_hf_home/hub"

try:
    print(f"    Trying snapshot dir: {model_dir}")
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True
    )
except Exception as e:
    print(f"    Snapshot failed ({e.__class__.__name__}), using repo_id with cache")
    processor = AutoProcessor.from_pretrained(
        repo_id,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=hub_cache,
    )

print(f"    ✓ Processor class: {processor.__class__.__name__}")
print(f"    ✓ Image processor patch size: {processor.image_processor.patch_size}")

img = Image.open(image_path)
print(f"\n[2] Test image: {image_path}")
print(f"    Image size (W x H): {img.size}")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},  # Use PIL image directly
        {"type": "text", "text": "Describe this image."}
    ]
}]

# Apply chat template (text only)
print(f"\n[3] Applying chat template...")
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
print(f"    Prompt length: {len(text)} chars")
print(f"    Prompt preview: {text[:200]}...")

# Process vision info (like vLLM does)
print(f"\n[4] Processing vision info...")
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    image_patch_size=processor.image_processor.patch_size,
    return_video_kwargs=True,
    return_video_metadata=True,
)

print(f"    image_inputs type: {type(image_inputs)}")
if isinstance(image_inputs, list) and image_inputs:
    print(f"    image_inputs[0] type: {type(image_inputs[0])}")
    if hasattr(image_inputs[0], 'size'):
        print(f"    image_inputs[0] size: {image_inputs[0].size}")

# Now process through the actual processor
print(f"\n[5] Processing through AutoProcessor...")
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print(f"\n[6] PROCESSOR OUTPUT ANALYSIS:")
print("-"*80)
print(f"Keys in processor output: {list(inputs.keys())}")
print(f"\ninput_ids shape: {inputs.input_ids.shape}")
print(f"  -> Number of text tokens: {inputs.input_ids.shape[1]}")

if hasattr(inputs, 'attention_mask'):
    print(f"\nattention_mask shape: {inputs.attention_mask.shape}")

if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
    print(f"\npixel_values shape: {inputs.pixel_values.shape}")
    print(f"  -> Batch size: {inputs.pixel_values.shape[0]}")
    print(f"  -> Channels: {inputs.pixel_values.shape[1]}")
    print(f"  -> Spatial dimensions: {inputs.pixel_values.shape[2:]}")

# THE KEY METADATA!
if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
    print(f"\n*** image_grid_thw: {inputs.image_grid_thw} ***")
    print(f"    Shape: {inputs.image_grid_thw.shape}")
    if inputs.image_grid_thw.numel() >= 3:
        grid = inputs.image_grid_thw[0] if inputs.image_grid_thw.dim() > 1 else inputs.image_grid_thw
        t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
        print(f"    Temporal (T): {t}")
        print(f"    Height tokens (H): {h}")
        print(f"    Width tokens (W): {w}")
        print(f"    Total vision tokens: {t * h * w}")
        
        # Calculate token→pixel mapping
        img_w, img_h = img.size
        patch_size = processor.image_processor.patch_size
        # Store these for later use
        actual_token_grid_w = w
        actual_token_grid_h = h
        
        print(f"\n[7] ACTUAL TOKEN→PIXEL MAPPING FROM PROCESSOR:")
        print(f"    Original image: {img_w}W x {img_h}H pixels")
        print(f"    Patch size: {patch_size}x{patch_size}")
        print(f"    Token grid: {w}W x {h}H")
        pixels_per_token_w = img_w / w
        pixels_per_token_h = img_h / h
        print(f"    Pixels per token: {pixels_per_token_w:.1f}W x {pixels_per_token_h:.1f}H")
        print(f"\n    ⚠ Note: Observed {pixels_per_token_w:.0f}×{pixels_per_token_h:.0f} px/token")
        print(f"           (Expected 28×28 for Qwen2.5-VL per documentation)")
        print(f"           Using ACTUAL values from processor!")

if hasattr(inputs, 'video_grid_thw') and inputs.video_grid_thw is not None:
    print(f"\nvideo_grid_thw: {inputs.video_grid_thw}")

# Check for position IDs or other spatial metadata
for key in inputs.keys():
    if 'pos' in key.lower() or 'grid' in key.lower():
        val = inputs[key]
        print(f"\n{key}: {val.shape if hasattr(val, 'shape') else val}")

print("\n" + "="*80)
print("SUMMARY - PIXEL TO TOKEN MAPPING")
print("="*80)

# Extract key info
patch_size = processor.image_processor.patch_size
img_w, img_h = img.size

# Qwen vision tokenization facts (from documentation):
# - Qwen2.5-VL: patch_size=14, but uses 2×2 patch merging → 28px per token
# - Qwen3-VL: patch_size=16, spatial compression → 32px per token
# - Images resized to multiples of the token block size

if patch_size == 14:
    # Qwen2.5-VL
    token_block_size = 28  # 2×2 merge of 14px patches
    model_name = "Qwen2.5-VL"
elif patch_size == 16:
    # Qwen3-VL  
    token_block_size = 32  # 2×2 grouping of 16px patches
    model_name = "Qwen3-VL"
else:
    # Unknown, use patch_size * 2 as estimate
    token_block_size = patch_size * 2
    model_name = f"Qwen-VL (patch_size={patch_size})"

print(f"\nModel: {model_name}")
print(f"Patch size: {patch_size}×{patch_size} pixels")
print(f"Token block size: {token_block_size}×{token_block_size} pixels")
print(f"  (Each vision token covers a {token_block_size}×{token_block_size} pixel region)")

# Calculate how image will be resized (to nearest multiple of token_block_size)
resized_w = ((img_w + token_block_size - 1) // token_block_size) * token_block_size
resized_h = ((img_h + token_block_size - 1) // token_block_size) * token_block_size

token_grid_w = resized_w // token_block_size
token_grid_h = resized_h // token_block_size

print(f"\nOriginal image: {img_w}W × {img_h}H pixels")
print(f"Resized to multiples of {token_block_size}: {resized_w}W × {resized_h}H pixels")
print(f"Token grid dimensions: {token_grid_w}W × {token_grid_h}H tokens")
print(f"Total vision tokens: {token_grid_w * token_grid_h}")

# Verify against image_grid_thw if available - USE ACTUAL VALUES!
if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
    grid = inputs.image_grid_thw[0] if inputs.image_grid_thw.dim() > 1 else inputs.image_grid_thw
    t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
    print(f"\nVerification from image_grid_thw: T={t}, H={h}, W={w}")
    
    # CRITICAL: Use ACTUAL observed values, not theoretical calculations
    if h == token_grid_h and w == token_grid_w:
        print("✓ MATCHES calculated token grid!")
    else:
        print(f"⚠ MISMATCH: Calculated {token_grid_w}×{token_grid_h}, got {w}×{h}")
        print(f"   → Using ACTUAL grid from processor: {w}W × {h}H")
        token_grid_w, token_grid_h = w, h
        
    # Update token block size based on ACTUAL processor behavior
    actual_block_w = img_w / w
    actual_block_h = img_h / h
    if abs(actual_block_w - actual_block_h) < 1:  # roughly square
        token_block_size = int(round((actual_block_w + actual_block_h) / 2))
        print(f"   → Actual token block size: ~{token_block_size}×{token_block_size} pixels")
    else:
        print(f"   → Non-square tokens: {actual_block_w:.1f}W × {actual_block_h:.1f}H pixels")
    
    resized_w = int(w * actual_block_w)
    resized_h = int(h * actual_block_h)

print("\n" + "-"*80)
print("PIXEL → TOKEN MAPPING FORMULA")
print("-"*80)
print(f"""
For your {img_w}×{img_h} image processed to {token_grid_w}×{token_grid_h} token grid:

SIMPLE METHOD (Recommended):
  1. Load mask: shape ({img_h}, {img_w}) boolean array
  2. Resize mask to ({token_grid_h}, {token_grid_w}) using NEAREST NEIGHBOR
  3. Each True value in resized mask = that token is in the gold region
  
PRECISE METHOD (if you need exact pixel→token mapping):
  - Pixel at (x, y) in ORIGINAL image
  - Token column: floor(x × {token_grid_w} / {img_w})
  - Token row: floor(y × {token_grid_h} / {img_h})
  - Clamp to [0, {token_grid_w}-1] for width, [0, {token_grid_h}-1] for height

Example Python code:
""")

print(f"""
from PIL import Image
import numpy as np

# Load pixel mask (H, W) boolean array
pixel_mask = np.load("nb_1_masks.npz")["masks"][0]  # shape: ({img_h}, {img_w})

# IMPORTANT: Transpose if mask is (W, H) instead of (H, W)
# Check: pixel_mask.shape should be (height={img_h}, width={img_w})
if pixel_mask.shape != ({img_h}, {img_w}):
    print(f"Warning: mask shape {{pixel_mask.shape}} != ({img_h}, {img_w})")
    
# Resize to token grid - NOTE: PIL expects (width, height)!
mask_img = Image.fromarray(pixel_mask.astype(np.uint8) * 255, mode='L')
token_mask_img = mask_img.resize(
    ({token_grid_w}, {token_grid_h}),  # (width, height) for PIL
    resample=Image.NEAREST  # Use NEAREST for boolean masks
)
token_mask = np.array(token_mask_img) > 0  # shape: ({token_grid_h}, {token_grid_w})

# Now token_mask[row, col] = True means vision token at that position is in gold region
# Use this mask to constrain attention during training
print(f"Token mask shape: {{token_mask.shape}}")
print(f"Gold region covers {{token_mask.sum()}} / {{token_mask.size}} tokens")
""")

print("\n" + "="*80)
print("NEXT STEPS FOR TRAINING")
print("="*80)
print("""
Now that you have the pixel→token mapping:

1. Convert all your gold region masks to token-level masks using the resize method above

2. During training, hook into the attention mechanism:
   - For vLLM: Modify attention masks in the forward pass
   - For HuggingFace: Use custom attention masks in model.forward()

3. Constrain cross-attention between text tokens and vision tokens:
   - Allow attention ONLY to vision tokens where token_mask = True
   - Zero out attention to masked-out regions (non-causal)

4. Training objective:
   - Standard VQA loss on gold regions only
   - Or: contrastive loss comparing full-image vs. gold-region-only answers

5. Implementation approach:
   - Start with HuggingFace transformers (easier to hook attention)
   - Use LoRA or full fine-tuning with attention constraints
   - Monitor: does model learn to rely only on gold regions?

Key challenge: vLLM is optimized for inference, not training.
Recommend using standard HuggingFace Trainer with custom attention masks.
""")