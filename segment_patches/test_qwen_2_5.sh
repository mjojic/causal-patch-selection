#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 test_qwen_2_5.py \
    --hf_home /mnt/shared/shared_hf_home \
    --model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct \
    --backend vllm --fp8 --tp 1 \
    --max_model_len 20000 \
    --gpu_memory_utilization 0.95 \
    --max_num_seqs 1 \
    --max_num_batched_tokens 8192