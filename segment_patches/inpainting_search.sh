# export HF_HOME=/mnt/shared/shared_hf_home

# export HF_HOME=/mnt/arc/mjojic/hf_overlay
# export HF_HUB_CACHE=/mnt/arc/mjojic/hf_overlay/hub
# export TRANSFORMERS_CACHE=/mnt/arc/mjojic/hf_overlay/transformers
# export DIFFUSERS_CACHE=/mnt/arc/mjojic/hf_overlay/diffusers

# export HF_DATASETS_CACHE=/mnt/shared/shared_hf_home/datasets
# export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE

python inpainting_search.py \
  --patch_dir naturalbench_objects \
  --output_dir inpainting_search_result \
  --vlm_device 1 \
  --inpaint_device 3 \
  --no_resize \
  --gpu_memory_utilization 0.7 \
  --max_model_len 15000 \
  --num_patches 5 \
  --min_sc_drop 0.02 \
  --num_samples_sc 25 \
  --use_qwen_diffusion_prompt \
  --start_idx 88 \
  --inpaint_model_id Manojb/stable-diffusion-2-1-base