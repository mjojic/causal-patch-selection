#!/usr/bin/env bash
# Aggregate consensus patches for seed bench.

python3 aggregate_patches_seed_bench.py \
  --vlm_device 2 --inpaint_device 2 \
  --inpaint_model_id /mnt/shared/shared_hf_home/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting \
  --qwen3_results_dir found_patches_seed_bench/models--Qwen--Qwen2.5-VL-72B-Instruct \
  --qwen25_results_dir found_patches_seed_bench/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
  --patch_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/seed_bench \
  --output_dir consensus_patches_seed_bench \
  --qwen3_model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.7 \
  --max_model_len 16000 \
  --use_qwen_diffusion_prompt \
  --num_samples_sc 10 \
  --start_idx 1745
