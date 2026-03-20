#!/bin/bash
# Test VLM inference only (no diffusion) for Qwen 2.5 VL 72B
# Use this to test memory consumption and FP4/FP8 quantization

# Default settings for Qwen 2.5 VL 72B
python3 test_vlm_only.py \
  --device 6 \
  --model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.7 \
  --max_model_len 9000 \
  --num_samples 5 \
  --num_examples 3 \
  --max_image_size 1024

# Memory optimization options to try if OOM:
# --gpu_memory_utilization 0.8    # Increase if stable, decrease if OOM
# --max_model_len 8000            # Reduce KV cache size (try 8000, 6000, 4000)
# --num_samples 3                 # Reduce batch size for self-consistency
# --max_image_size 768            # Reduce image size (fewer vision tokens)
