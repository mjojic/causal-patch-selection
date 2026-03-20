#!/bin/bash
# Clean v4 training script

python attention_alignment_v4.py \
    --gpu 3 \
    --model_path /mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct \
    --consensus_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/consensus_search/consensus_patches \
    --batch_size 25 \
    --lr 2e-4 \
    --n_epochs 5 \
    --grad_accum 1 \
    --lambda_attn 25.0 \
    --save_dir ./attn_align_first_token_5_epoch \
    --quantization 4bit \
    --loss_func attention_alignment \
    --log_every 1 \
    --tqdm \
    --txt_token_for_attn first
