python attention_alignment.py \
    --gpu 2 \
    --attn_implementation flash_attention_2 \
    --model_path /mnt/shared/shared_hf_home/models--Qwen--Qwen3-VL-8B-Instruct \
    --consensus_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/consensus_search/consensus_patches \
    --batch_size 5 \
    --num_workers 0 \
    --lr 2e-4 \
    --weight_decay 0.0 \
    --n_epochs 3 \
    --grad_accum 1 \
    --lambda_attn 25.0 \
    --log_every 1 \
    --save_dir ./tuned_attn_align_qwen3_vl_8b \
    --quantization 4bit \
    --no_ckpt \
    --loss_func attention_alignment

