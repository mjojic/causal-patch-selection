# pixel_to_token_utils.py

from PIL import Image
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List

# -----------------------------
# Vision span helper (BUGFIX)
# -----------------------------
def find_vision_spans(input_ids_1d: torch.Tensor, model) -> List[Tuple[int, int]]:
    """
    Find [vision_start, vision_end] spans in a single sequence of input_ids.
    Returns list of (start_index, end_index) positions (inclusive markers).
    Image tokens are positions (vs+1 ... ve-1).
    """
    vs_id = model.config.vision_start_token_id
    ve_id = model.config.vision_end_token_id

    vs_pos = (input_ids_1d == vs_id).nonzero(as_tuple=False).flatten().tolist()
    ve_pos = (input_ids_1d == ve_id).nonzero(as_tuple=False).flatten().tolist()

    spans = []
    if not vs_pos or not ve_pos:
        return spans

    j = 0
    for s in vs_pos:
        while j < len(ve_pos) and ve_pos[j] < s:
            j += 1
        if j < len(ve_pos):
            spans.append((s, ve_pos[j]))
            j += 1
    return spans


def _get_merge_size(model, processor) -> int:
    """
    Qwen-VL stacks often expose either:
      - model.config.spatial_merge_size
      - processor.image_processor.merge_size
    Fallback to 2 (common default for Qwen-VL).
    """
    ms = getattr(getattr(model, "config", None), "spatial_merge_size", None)
    if ms is None:
        ms = getattr(getattr(processor, "image_processor", None), "merge_size", None)
    if ms is None:
        ms = 2
    return int(ms)


def llm_image_token_grid_from_inputs(
    inputs: Dict[str, torch.Tensor],
    model,
    processor,
) -> Tuple[int, int, int]:
    """
    Compute (t, h_llm, w_llm) for the LLM-side image tokens.

    inputs["image_grid_thw"][0] gives the *pre-merge* grid (t,h_patch,w_patch).
    The LLM sees h_llm = h_patch//merge, w_llm = w_patch//merge.
    """
    if "image_grid_thw" not in inputs:
        raise KeyError("inputs is missing 'image_grid_thw' (needed to infer token grid).")

    t, h_patch, w_patch = inputs["image_grid_thw"][0].tolist()
    merge = _get_merge_size(model, processor)

    if h_patch % merge != 0 or w_patch % merge != 0:
        raise RuntimeError(f"image_grid_thw ({h_patch},{w_patch}) not divisible by merge_size={merge}")

    return int(t), int(h_patch // merge), int(w_patch // merge)


# -----------------------------
# PNG loading
# -----------------------------
def _load_binary_png_mask(png_path: str, threshold: int = 127) -> np.ndarray:
    """
    Load a BW/grayscale PNG mask into boolean array (H,W).
    Pixels > threshold are True.
    """
    m = Image.open(png_path).convert("L")
    arr = np.array(m)
    return (arr > threshold)


# -----------------------------
# Simple resize (your old method)
# -----------------------------
def pixel_mask_to_token_mask(
    pixel_mask_hw: np.ndarray,
    token_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a boolean pixel mask (H,W) -> boolean token mask (h_llm,w_llm) using NN.
    This is an approximation if the model preprocessing resizes/pads/crops differently.
    """
    token_h, token_w = token_hw
    mask_img = Image.fromarray(pixel_mask_hw.astype(np.uint8) * 255, mode="L")
    token_mask_img = mask_img.resize((token_w, token_h), resample=Image.NEAREST)
    token_mask = (np.array(token_mask_img) > 0)
    return token_mask


# -----------------------------
# More correct conversion using image_grid_thw + patch_size + merge
# -----------------------------
def pixel_mask_to_llm_token_mask_from_inputs(
    pixel_mask_hw: np.ndarray,
    inputs: Dict[str, torch.Tensor],
    model,
    processor,
    image_size_wh: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convert pixel mask (H,W in *original image space*) into LLM token mask (h_llm,w_llm)
    using the model token grid geometry.

    Steps:
      1) Optional: NN-resize mask to match image_size_wh (W,H) if provided and differs.
      2) Use inputs["image_grid_thw"] = (t, h_patch, w_patch) and patch_size to define
         a *processed* pixel canvas of (H_proc = h_patch*patch_size, W_proc = w_patch*patch_size).
      3) NN-resize mask to (W_proc,H_proc).
      4) Pool into patch grid (h_patch,w_patch) via ANY over each patch_size x patch_size block.
      5) Apply spatial merge (merge x merge) via ANY to produce (h_llm,w_llm).

    This still assumes the model “content” maps to that processed canvas without exotic cropping,
    but it respects the token grid much better than direct resize-to-(h_llm,w_llm).
    """
    if "image_grid_thw" not in inputs:
        raise KeyError("inputs missing image_grid_thw")

    # Step 1: align mask to original image size if you want
    px = pixel_mask_hw.astype(bool)
    if image_size_wh is not None:
        W, H = image_size_wh
        if px.shape != (H, W):
            px_img = Image.fromarray(px.astype(np.uint8) * 255, mode="L")
            px_img = px_img.resize((W, H), resample=Image.NEAREST)
            px = (np.array(px_img) > 0)

    t, h_patch, w_patch = inputs["image_grid_thw"][0].tolist()
    if int(t) != 1:
        raise RuntimeError(f"Expected t=1 for static images; got t={t}")

    patch = int(getattr(processor.image_processor, "patch_size", 14))
    merge = _get_merge_size(model, processor)

    H_proc = int(h_patch) * patch
    W_proc = int(w_patch) * patch

    # Step 3: resize to processed pixel canvas
    px_img = Image.fromarray(px.astype(np.uint8) * 255, mode="L")
    px_proc = px_img.resize((W_proc, H_proc), resample=Image.NEAREST)
    px_proc = (np.array(px_proc) > 0)  # (H_proc, W_proc)

    # Step 4: pool to patch grid (ANY within each patch block)
    # reshape: (h_patch, patch, w_patch, patch) -> (h_patch,w_patch)
    px_proc = px_proc.reshape(int(h_patch), patch, int(w_patch), patch)
    patch_mask = px_proc.any(axis=(1, 3))  # (h_patch, w_patch)

    # Step 5: apply merge pooling to get LLM token grid
    if int(h_patch) % merge != 0 or int(w_patch) % merge != 0:
        raise RuntimeError("patch grid not divisible by merge")

    h_llm = int(h_patch) // merge
    w_llm = int(w_patch) // merge

    patch_mask = patch_mask.reshape(h_llm, merge, w_llm, merge)
    llm_mask = patch_mask.any(axis=(1, 3))  # (h_llm,w_llm)
    return llm_mask


def png_pixel_mask_to_token_mask(
    png_mask_path: str,
    inputs: Dict[str, torch.Tensor],
    model,
    processor,
    image_size_wh: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    threshold: int = 127,
    use_grid_aware: bool = True,
) -> torch.Tensor:
    """
    End-to-end convenience:
      PNG mask -> token mask aligned to the model's LLM image-token grid.
    """
    px = _load_binary_png_mask(png_mask_path, threshold=threshold)  # (H,W) bool

    if use_grid_aware:
        tok = pixel_mask_to_llm_token_mask_from_inputs(
            px, inputs=inputs, model=model, processor=processor, image_size_wh=image_size_wh
        )
    else:
        # old approximate method: infer (h_llm,w_llm) and resize directly
        t, h_llm, w_llm = llm_image_token_grid_from_inputs(inputs, model, processor)
        if t != 1:
            raise RuntimeError(f"Expected t=1 for image, got t={t}")
        tok = pixel_mask_to_token_mask(px, (h_llm, w_llm))

    tok_t = torch.from_numpy(tok).to(dtype=torch.bool)
    if device is not None:
        tok_t = tok_t.to(device)
    return tok_t


def verify_token_mask_matches_vision_span(
    token_mask_hw: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    model,
) -> None:
    """
    Sanity check: flattened token mask length matches number of image tokens
    in the input_ids vision span.
    """
    spans = find_vision_spans(inputs["input_ids"][0], model)
    if not spans:
        raise RuntimeError("No vision span found in input_ids; cannot verify.")
    vs, ve = spans[0]
    num_img_tokens = ve - (vs + 1)
    if token_mask_hw.numel() != num_img_tokens:
        raise RuntimeError(
            f"Token-mask size mismatch: token_mask_hw has {token_mask_hw.numel()} elems "
            f"but vision span implies {num_img_tokens} image tokens (vs={vs}, ve={ve})."
        )