python train_attn.py \
    --gpu 2 \
    --consensus_dir /mnt/arc/mjojic/causal-patch-selection/segment_patches/consensus_search/consensus_patches \
    --batch_size 32 \
    --lr 2e-5 \
    --n_epochs 3 \
    --save_dir ./checkpoint_full