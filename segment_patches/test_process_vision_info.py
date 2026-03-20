#!/usr/bin/env python3
"""
Test script to examine process_vision_info() output structure
"""
import sys
from PIL import Image
import numpy as np

print("Testing process_vision_info() from qwen_vl_utils...")
print("="*60)

try:
    from qwen_vl_utils import process_vision_info
    print("✓ Successfully imported process_vision_info")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("\nTo install: pip install qwen-vl-utils")
    sys.exit(1)

# Test with a sample image
image_path = "/mnt/arc/mjojic/causal-patch-selection/segment_patches/found_patches/seg_comparison_1.png"
img = Image.open(image_path)
print(f"\nTest image: {image_path}")
print(f"Image size: {img.size} (W x H)")
print(f"Image mode: {img.mode}")

# Create messages in Qwen format
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": "Describe this."}
    ]
}]

print("\n" + "="*60)
print("Calling process_vision_info()...")
print("="*60)

# Test with default settings (Qwen2.5-VL: patch_size=14)
print("\n1. DEFAULT SETTINGS (patch_size=14, for Qwen2.5-VL):")
try:
    result = process_vision_info(messages, image_patch_size=14)
    
    if isinstance(result, tuple):
        print(f"   Returns tuple with {len(result)} elements")
        
        image_inputs, video_inputs = result[:2]
        
        print(f"\n   image_inputs type: {type(image_inputs)}")
        if image_inputs is not None:
            if hasattr(image_inputs, 'shape'):
                print(f"   image_inputs shape: {image_inputs.shape}")
                print(f"   image_inputs dtype: {image_inputs.dtype}")
            elif isinstance(image_inputs, list):
                print(f"   image_inputs is list with {len(image_inputs)} items")
                for i, item in enumerate(image_inputs[:3]):  # Show first 3
                    print(f"     Item {i}: {type(item)} - {item.shape if hasattr(item, 'shape') else 'N/A'}")
        
        print(f"\n   video_inputs type: {type(video_inputs)}")
        if video_inputs is not None:
            print(f"   video_inputs: {video_inputs}")
    else:
        print(f"   Unexpected return type: {type(result)}")
        
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-"*60)

# Test with video kwargs (for newer versions)
print("\n2. WITH return_video_kwargs=True:")
try:
    result = process_vision_info(
        messages,
        image_patch_size=14,
        return_video_kwargs=True
    )
    
    print(f"   Returns {len(result)} elements")
    if len(result) >= 3:
        image_inputs, video_inputs, video_kwargs = result[:3]
        print(f"   video_kwargs: {video_kwargs}")
        
except Exception as e:
    print(f"   Not supported or error: {e}")

print("\n" + "-"*60)

# Test with Qwen3-VL settings (patch_size=16)
print("\n3. QWEN3-VL SETTINGS (patch_size=16):")
try:
    result = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    print(f"   Returns {len(result)} elements")
    if len(result) >= 3:
        image_inputs, video_inputs, video_kwargs = result[:3]
        
        if image_inputs is not None and hasattr(image_inputs, 'shape'):
            print(f"   image_inputs shape: {image_inputs.shape}")
            
            # Try to infer grid dimensions
            num_tokens = image_inputs.shape[0] if len(image_inputs.shape) > 0 else 0
            print(f"   Total visual tokens: {num_tokens}")
            
            # Estimate grid based on image size and patch size
            w, h = img.size
            est_grid_w = w // 16
            est_grid_h = h // 16
            print(f"   Estimated grid (W x H): {est_grid_w} x {est_grid_h} = {est_grid_w * est_grid_h} tokens")
            
except Exception as e:
    print(f"   Not supported or error: {e}")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

# Calculate expected token counts for different resolutions
w, h = img.size
print(f"\nOriginal image: {w}W x {h}H")

for patch_size in [14, 16]:
    print(f"\nPatch size {patch_size}x{patch_size}:")
    
    # Naive calculation (before any dynamic resolution adjustments)
    naive_w = w // patch_size
    naive_h = h // patch_size
    naive_tokens = naive_w * naive_h
    print(f"  Naive grid: {naive_w}W x {naive_h}H = {naive_tokens} tokens")
    
    # With padding to multiples
    pad_w = ((w + patch_size - 1) // patch_size) * patch_size
    pad_h = ((h + patch_size - 1) // patch_size) * patch_size
    pad_tokens = (pad_w // patch_size) * (pad_h // patch_size)
    print(f"  With padding: {pad_w // patch_size}W x {pad_h // patch_size}H = {pad_tokens} tokens")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("""
To map pixel masks to vision tokens:

1. Determine actual token grid dimensions from process_vision_info() output
2. Resize/pad your boolean mask to match the token grid
3. Use this token-level mask to constrain attention

Key questions to answer:
- Does the output include spatial metadata (grid dimensions)?
- Is there a deterministic mapping from image patches to tokens?
- Do we need to account for ViT's [CLS] token or other special tokens?
""")
