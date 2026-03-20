#!/bin/bash

python evaluate_accuracy.py \
  --base_model_dir /mnt/arc/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct \
  --lora_dir /mnt/arc/mjojic/causal-patch-selection/training/lora_rs_first_token_b30 \
  --start_idx 3000 \
  --end_idx 3150 \
  --num_samples_sc 10 \
  --temperature 0.7 \
  --top_p 0.9 \
  --gpu 1 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.35 \
  --max_model_len 10000 \
  --output_json accuracy_results_vqa_only.json \
  --merge_weights
