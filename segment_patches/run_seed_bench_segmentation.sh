#!/usr/bin/env bash
# Run seed bench segmentation on GPU 3.

export CUDA_VISIBLE_DEVICES=3
python3 "$(dirname -- "$0")/segment_seed_bench.py" "$@"
