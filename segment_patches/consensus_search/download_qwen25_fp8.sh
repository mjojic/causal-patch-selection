#!/usr/bin/env bash
set -euo pipefail

REPO_ID="RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic"
CACHE_DIR="/mnt/shared/shared_hf_home/hub"

echo "Downloading ${REPO_ID} to ${CACHE_DIR} ..."

conda run -n qwen3-vllm \
  huggingface-cli download "$REPO_ID" \
    --cache-dir "$CACHE_DIR"

echo "Done. Model stored under:"
echo "  ${CACHE_DIR}/models--RedHatAI--Qwen2.5-VL-72B-Instruct-FP8-dynamic"
