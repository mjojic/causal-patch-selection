#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test SD1.5 inpainting: make a fully blue image and inpaint a fluffy white cloud.

Run:
  CUDA_VISIBLE_DEVICES=3 python test_sd15_inpaint_cloud.py
"""

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline


def load_inpaint_pipe(model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    return pipe, device


def main():
    pipe, device = load_inpaint_pipe()

    # Make a non-square blue image (to confirm pipeline respects aspect ratio)
    W, H = 768, 512
    img = Image.new("RGB", (W, H), (40, 120, 255))  # blue sky

    # Mask region to inpaint (white=edit, black=keep)
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([int(0.20 * W), int(0.15 * H), int(0.80 * W), int(0.70 * H)], fill=255)

    prompt = "A single fluffy white cloud in a blue sky, realistic, soft edges."
    negative = "text, watermark, logo, artifacts, blur, distortion"

    # Use deterministic seed to make debugging easier
    generator = torch.Generator(device=device).manual_seed(0) if device == "cuda" else None

    # SD1.5 inpaint expects height/width multiples of 8 (these are)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=img,
            mask_image=mask,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=1.0,      # force strong edit inside mask
            generator=generator,
        ).images[0]

    out_path = "sd15_cloud_inpaint_test.png"
    out.save(out_path)

    # Save inputs for sanity
    img.save("sd15_cloud_input.png")
    mask.save("sd15_cloud_mask.png")

    print("Device:", device)
    print("Saved:", out_path)
    print("Also saved: sd15_cloud_input.png, sd15_cloud_mask.png")


if __name__ == "__main__":
    main()
