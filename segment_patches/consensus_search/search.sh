# python3 search.py \
#   --vlm_device 0 --inpaint_device 1 \
#   --patch_dir ../naturalbench_objects \
#   --output_dir found_patches \
#   --vlm_model_dir /mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
#   --tensor_parallel_size 1 \
#   --gpu_memory_utilization 0.65 \
#   --max_model_len 10000 \
#   --start_idx 5204 --end_idx 6000 \
#   --min_sc_drop 0.1 \
#   --num_patches 5 \
#   --no_resize --use_qwen_diffusion_prompt


# Qwen 2.5 VL 72B: search.py uses enforce_eager=True (standard attention)
python3 search.py \
  --vlm_device 3 --inpaint_device 1 \
  --patch_dir ../naturalbench_objects \
  --output_dir found_patches \
  --vlm_model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-mjojic \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.8 \
  --max_model_len 10000 \
  --start_idx 4069 --end_idx 6000 \
  --min_sc_drop 0.1 \
  --num_patches 5 \
  --no_resize --use_qwen_diffusion_prompt

# python3 aggregate_patches.py \
#   --vlm_device 0 --inpaint_device 1 \
#   --qwen3_results_dir found_patches/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
#   --qwen25_results_dir found_patches/models--Qwen--Qwen2.5-VL-72B-Instruct-mjojic \
#   --patch_dir ../naturalbench_objects \
#   --output_dir consensus_patches \
#   --qwen3_model_dir /mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-32B-Instruct-FP8 \
#   --tensor_parallel_size 1 \
#   --gpu_memory_utilization 0.8 \
#   --max_model_len 16000 \
#   --use_qwen_diffusion_prompt \
#   --num_samples_sc 15 \
#   --start_idx 5193 --end_idx 6000 

# python3 search.py \
#   --vlm_device 2 --inpaint_device 3 \
#   --patch_dir ../naturalbench_objects \
#   --output_dir found_patches \
#   --vlm_model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-mjojic \
#   --tensor_parallel_size 1 \
#   --gpu_memory_utilization 0.85 \
#   --max_model_len 9000 \
#   --start_idx 1621 --end_idx 2253 \
#   --min_sc_drop 0.1 \
#   --num_patches 5 \
#   --no_resize --use_qwen_diffusion_prompt

