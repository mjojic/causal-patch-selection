#!/bin/bash

python evaluate_attention_distances.py \
    --gpu 1 \
    --attn_implementation flash_attention_2 \
    --base_model_path /mnt/arc/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct \
    --finetuned_model_path ./attn_align_bs1_q4_fa2 \
    --consensus_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/consensus_search/consensus_patches \
    --output_csv attention_distances.csv \
    --output_plot attention_distances.png \
    --batch_size 1 \
    --num_workers 0 \
    --quantization 4bit
