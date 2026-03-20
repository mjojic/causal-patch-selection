#!/bin/bash
# Rejection Sampling + Attention Alignment Training Script

python rs_attention_alignment.py \
    --gpu 3 \
    --model_path /mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct \
    --consensus_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/consensus_search/consensus_patches \
    --batch_size 45 \
    --lr 2e-4 \
    --n_epochs 3 \
    --grad_accum 1 \
    --lambda_attn 25.0 \
    --num_samples 1 \
    --max_new_tokens 128 \
    --temperature 0.7 \
    --save_dir ./lora_rs_first_token_b45 \
    --quantization 4bit \
    --loss_func attention_alignment \
    --log_every 1 \
    --tqdm \
    --txt_token_for_attn first \
    --batch_generations
