export CUDA_VISIBLE_DEVICES=3

python segment_naturalbench.py \
  --output_dir naturalbench_objects \
  --device cuda:0 \
  --start_idx 2254 \
  --end_idx 7600 \
  --min_area 50
