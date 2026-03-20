#!/usr/bin/env python3
"""
Download Qwen2.5-VL-72B-Instruct into the shared HF home under a distinct directory
(models--Qwen--Qwen2.5-VL-72B-Instruct-mjojic) so it doesn't conflict with the
existing permission-locked copy. Use the resulting path as --vlm_model_dir, e.g.:

  --vlm_model_dir /mnt/shared/shared_hf_home/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-mjojic
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
# Directory name under shared HF hub so we have write access (slightly different name)
HF_HUB = os.environ.get("HF_HOME", "/mnt/shared/shared_hf_home")
HUB_DIR = Path(HF_HUB) / "hub"
LOCAL_DIR_NAME = "models--Qwen--Qwen2.5-VL-72B-Instruct-mjojic"
LOCAL_DIR = HUB_DIR / LOCAL_DIR_NAME


def main() -> None:
    print(f"[download] repo: {REPO_ID}")
    print(f"[download] destination: {LOCAL_DIR}")
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"[download] done. Use with:")
    print(f"  --vlm_model_dir {LOCAL_DIR}")


if __name__ == "__main__":
    main()
