python evaluate_pope_hf.py \
    --gpu 4 \
    --base_model_dir "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct" \
    --lora_vqa_attn_dir "/mnt/arc/mjojic/causal-patch-selection/training/lora_v4_attn_align" \
    --pope_cache_dir "/mnt/shared/shared_hf_home/datasets" \
    --start_idx 0 \
    --end_idx 915 \
    --max_tokens 1024 \
    --output_json "./pope_eval_results/pope_results_hf_lora_v4_attn_align.json" \
    --eval_base
