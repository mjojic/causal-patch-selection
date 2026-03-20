#!/usr/bin/env python3
"""Download Qwen3-VL-8B-Instruct to the shared HF cache."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    cache_dir="/mnt/shared/shared_hf_home/hub",
)
