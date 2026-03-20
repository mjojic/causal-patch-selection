#!/usr/bin/env bash
# Download Qwen3-VL-32B-Instruct-FP8 to shared HF home.

set -e
REPO_ID="Qwen/Qwen3-VL-32B-Instruct-FP8"
TARGET_DIR="/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8"

mkdir -p "$TARGET_DIR"
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=\"$REPO_ID\",
    local_dir=\"$TARGET_DIR\",
    local_dir_use_symlinks=False,
)
print('Downloaded to', '$TARGET_DIR')
"
