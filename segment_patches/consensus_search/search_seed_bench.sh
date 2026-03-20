#!/usr/bin/env bash
# Seed bench consensus search: GPU 1 for both VLM and inpainting, utilization 0.65.

# python3 search_seed_bench.py \
#   --vlm_device 2 --inpaint_device 2 \
#   --patch_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/seed_bench \
#   --output_dir found_patches_seed_bench \
#   --vlm_model_dir /mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
#   --tensor_parallel_size 1 \
#   --gpu_memory_utilization 0.65 \
#   --max_model_len 10000 \
#   --min_sc_drop 0.1 \
#   --num_patches 5 \
#   --no_resize --use_qwen_diffusion_prompt \
#   --start_idx 2310 --end_idx 3000

# --vlm_model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct \

# python3 search_seed_bench.py \
#   --vlm_device 2 --inpaint_device 2 \
#   --patch_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/seed_bench \
#   --output_dir found_patches_seed_bench \
#   --vlm_model_dir /mnt/shared/shared_hf_home/hub/models--RedHatAI--Qwen2.5-VL-72B-Instruct-FP8-dynamic \
#   --tensor_parallel_size 1 \
#   --gpu_memory_utilization 0.7 \
#   --max_model_len 8000 \
#   --min_sc_drop 0.1 \
#   --num_patches 5 \
#   --no_resize --use_qwen_diffusion_prompt \
#   --start_idx 2315 --end_idx 3000

python3 aggregate_patches_seed_bench.py \
  --vlm_device 2 --inpaint_device 2 \
  --inpaint_model_id /mnt/shared/shared_hf_home/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting \
  --qwen3_results_dir found_patches_seed_bench/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
  --qwen25_results_dir found_patches_seed_bench/models--RedHatAI--Qwen2.5-VL-72B-Instruct-FP8-dynamic \
  --patch_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/seed_bench \
  --output_dir consensus_patches_seed_bench \
  --qwen3_model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.5 \
  --max_model_len 8000 \
  --use_qwen_diffusion_prompt \
  --num_samples_sc 10 \
  --start_idx 2814
