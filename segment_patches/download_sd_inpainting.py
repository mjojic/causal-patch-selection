#!/usr/bin/env python3
"""
Download the Stable Diffusion v1.5 inpainting model for use with diffusers.

Saves to a single directory containing model_index.json and weights so that
StableDiffusionInpaintPipeline.from_pretrained(output_dir) works offline.

Usage:
  python download_sd_inpainting.py [--output_dir DIR]
  # Then use in aggregate_patches.sh:
  #   --inpaint_model_id /path/to/output_dir
"""
import argparse
import os


REPO_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_OUTPUT_DIR = "/mnt/shared/shared_hf_home/stable-diffusion-inpainting"


def main():
    parser = argparse.ArgumentParser(
        description="Download Stable Diffusion v1.5 inpainting model for diffusers."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=REPO_ID,
        help=f"HuggingFace repo id (default: {REPO_ID})",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo_id} to {output_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Done. Use this path for --inpaint_model_id:\n  {output_dir}")


if __name__ == "__main__":
    main()
