#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import datasets
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------
# Environment defaults for vLLM
# ---------------------------------------------------------------------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Core parameters
    parser.add_argument(
        "--patch_dir",
        type=str,
        required=True,
        help="Directory where segmentation masks were saved "
             "(nb_{idx}_masks.npz etc.)",
    )
    parser.add_argument(
        "--num_samples_sc",
        type=int,
        default=15,
        help="Number of samples for self-consistency evaluation",
    )
    parser.add_argument(
        "--min_sc_drop",
        type=float,
        default=0.0,
        help="Minimum SC drop w.r.t original answer to select a patch",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct-FP8",
        help="Qwen3-VL checkpoint (FP8 recommended for memory)",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=1,
        help="Maximum number of segmentation patches to select",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./naturalbench_seg_patch_search_results_vllm",
        help="Directory to save results",
    )
    parser.add_argument("--split", type=str, default="test")

    # NEW: dataset index range
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Inclusive start index of NaturalBench to process",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Exclusive end index of NaturalBench to process (-1 = until end)",
    )

    # vLLM-specific knobs
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size used by vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory vLLM is allowed to pre-allocate",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Max model length for vLLM (affects KV cache memory)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for self-consistency",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for vLLM",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Dataset loading – NaturalBench (lmms-eval format)
# ---------------------------------------------------------------------
def load_naturalbench_dataset(split: str = "test"):
    ds = datasets.load_dataset("BaiqiL/NaturalBench-lmms-eval", split=split)
    return ds


def datapoint_to_dict(dp):
    """
    Convert a NaturalBench-lmms-eval row into the format expected by the
    patch-search pipeline:

    {
      "image": PIL.Image,
      "question": str,
      "answer_choices": [str, ...],
      "gold_answer": str,
      "gold_letter": 'A'/'B'/...,
      "spurious_attribute": arbitrary metadata (here: Question_Type)
    }

    Handling:
      - yes_no:
          Question_Type == "yes_no"
          Answer in {"Yes","No"}
          Choices: [Yes, No]; gold_letter: A/B
      - multiple_choice:
          Question_Type == "multiple_choice"
          Question string often includes options like:
            "Option: A:Gray; B:Blue;"
          or
            "...? A. Foo B. Bar"
          We parse options when possible; otherwise fallback to generic labels.
    """
    img = dp["Image"]
    question_raw = dp["Question"]
    q_type = dp["Question_Type"]
    ans_raw = str(dp["Answer"]).strip()

    # Default values
    base_question = question_raw
    answer_choices: List[str] = []
    gold_letter: Optional[str] = None
    gold_answer: Optional[str] = None

    if q_type == "yes_no":
        # Normalize yes/no
        answer_choices = ["Yes", "No"]
        if ans_raw.lower().startswith("y"):
            gold_letter = "A"
            gold_answer = "Yes"
        else:
            gold_letter = "B"
            gold_answer = "No"

    else:  # multiple_choice
        # Try to parse "Option: A:...; B:...;"
        optA, optB = "Option A", "Option B"
        m = re.search(r"Option:\s*A:(.*?);\s*B:(.*?)(?:;|$)", question_raw)
        if m:
            optA = m.group(1).strip()
            optB = m.group(2).strip()
            base_question = question_raw.split("Option:")[0].strip()
        else:
            # Try "... A. ... B. ..."
            m2 = re.search(r"\bA\.\s*(.*?)\s*B\.\s*(.*)$", question_raw)
            if m2:
                optA = m2.group(1).strip()
                optB = m2.group(2).strip()
                # crude split to remove the trailing options from question
                base_question = question_raw.split("A.")[0].strip()

        answer_choices = [optA, optB]

        # NaturalBench-lmms-eval stores the correct choice as 'A' or 'B'
        gold_letter = ans_raw.upper()
        idx = ord(gold_letter) - ord("A")
        if 0 <= idx < len(answer_choices):
            gold_answer = answer_choices[idx]
        else:
            gold_answer = ans_raw

    return {
        "image": img,
        "question": base_question,
        "answer_choices": answer_choices,
        "gold_answer": gold_answer,
        "gold_letter": gold_letter,
        # For compatibility with your JSON schema; NaturalBench doesn't
        # have "spurious" tags, so we just record Question_Type here.
        "spurious_attribute": q_type,
    }


# ---------------------------------------------------------------------
# Image utilities for binary masks
# ---------------------------------------------------------------------
def apply_binary_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply a boolean mask (H, W) to the image: masked pixels become black.
    """
    img_np = np.array(image).copy()  # H, W, 3

    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D boolean array (H, W).")
    if mask.shape != img_np.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image spatial shape {img_np.shape[:2]}"
        )

    # Boolean index on first two dims; applies to all channels
    img_np[mask] = 0
    return Image.fromarray(img_np)


# ---------------------------------------------------------------------
# Prompting & parsing
# ---------------------------------------------------------------------
def make_text_prompt(question: str, answer_choices: List[str]) -> str:
    prompt = (
        "Given the following question, and answer choices, do the following:\n"
        "1) Think step by step, first identify what you can see in the image "
        "that is relevant to the question.\n"
        "2) Based on this information, reason about each answer choice and "
        "how likely it is to be correct.\n"
        "3) Finally, select the index of the single best answer choice.\n\n"
    )
    prompt += f"Question: {question}\n"
    prompt += "Answer Choices:\n"
    options = ["A", "B", "C", "D", "E"]
    for idx, choice in enumerate(answer_choices):
        prompt += f"{options[idx]}: {choice}\n"
    prompt += (
        "\nPlease provide the final answer in this JSON format: "
        "{'final_answer_letter': <letter>}\n"
    )
    return prompt


def parse_qwen_output(text: str) -> Optional[str]:
    m = re.search(r"\{.*?final_answer_letter.*?\}", text, re.DOTALL)
    if m:
        json_str = m.group(0)
        json_str_norm = json_str.replace("'", '"')
        json_str_norm = re.sub(r",\s*}", "}", json_str_norm)
        try:
            obj = json.loads(json_str_norm)
            if "final_answer_letter" in obj:
                return obj["final_answer_letter"].strip().upper()
        except Exception:
            pass

    m2 = re.search(r"final_answer_letter\s*[:=]\s*['\"]?([A-Da-d])['\"]?", text)
    if m2:
        return m2.group(1).upper()

    m3 = re.search(r"^\s*([A-D])\s*$", text, re.MULTILINE)
    if m3:
        return m3.group(1).upper()

    return None


# ---------------------------------------------------------------------
# vLLM setup and multimodal helpers
# ---------------------------------------------------------------------
def init_vllm_engine(
    model_id: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
) -> LLM:
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=max_model_len,
        seed=seed,
    )
    return llm


def prepare_inputs_for_vllm(
    messages: List[Dict[str, Any]], processor: AutoProcessor
) -> Dict[str, Any]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


# ---------------------------------------------------------------------
# Self-consistency with vLLM (single datapoint)
# ---------------------------------------------------------------------
def self_consistency_single(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    question = make_text_prompt(data_dict["question"], data_dict["answer_choices"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_dict["image"]},
                {"type": "text", "text": question},
            ],
        }
    ]
    vllm_input = prepare_inputs_for_vllm(messages, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples_sc,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    output = outputs[0]

    num_extracted = 0
    letter_counts: Dict[str, int] = {}
    gold_letter = data_dict["gold_letter"]

    for out in output.outputs:
        text = out.text
        parsed_letter = parse_qwen_output(text)
        if parsed_letter is not None:
            num_extracted += 1
            letter_counts[parsed_letter] = letter_counts.get(parsed_letter, 0) + 1

    majority_letter = (
        max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
    )

    if num_extracted > 0 and gold_letter is not None:
        num_gold = letter_counts.get(gold_letter, 0)
        gold_sc = num_gold / num_extracted
    else:
        gold_sc = None

    return {
        "num_extracted": num_extracted,
        "letter_counts": letter_counts,
        "majority_letter": majority_letter,
        "gold_sc": gold_sc,
    }

def self_consistency_batch(
    llm: LLM,
    processor: AutoProcessor,
    data_dicts: List[Dict[str, Any]],
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Batched self-consistency over multiple (image, question, choices).

    Returns a list of:
      {
        "num_extracted": int,
        "letter_counts": dict,
        "majority_letter": str or None,
        "gold_sc": float or None,
      }
    """
    if not data_dicts:
        return []

    vllm_inputs = []
    gold_letters: List[Optional[str]] = []

    for dp in data_dicts:
        question = make_text_prompt(dp["question"], dp["answer_choices"])
        gold_letters.append(dp["gold_letter"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dp["image"]},
                    {"type": "text", "text": question},
                ],
            }
        ]
        vllm_inputs.append(prepare_inputs_for_vllm(messages, processor))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples_sc,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    all_results: List[Dict[str, Any]] = []
    for gold_letter, output in zip(gold_letters, outputs):
        num_extracted = 0
        letter_counts: Dict[str, int] = {}

        for out in output.outputs:
            text = out.text
            parsed_letter = parse_qwen_output(text)
            if parsed_letter is not None:
                num_extracted += 1
                letter_counts[parsed_letter] = letter_counts.get(parsed_letter, 0) + 1

        majority_letter = (
            max(letter_counts.items(), key=lambda x: x[1])[0]
            if letter_counts
            else None
        )

        # SC wrt gold
        if num_extracted > 0 and gold_letter is not None:
            num_gold = letter_counts.get(gold_letter, 0)
            gold_sc = num_gold / num_extracted
        else:
            gold_sc = None

        all_results.append(
            {
                "num_extracted": num_extracted,
                "letter_counts": letter_counts,
                "majority_letter": majority_letter,
                "gold_sc": gold_sc,
            }
        )

    return all_results



# ---------------------------------------------------------------------
# SC helper: compute SC wrt arbitrary letter
# ---------------------------------------------------------------------
def sc_wrt_letter(
    counts: Dict[str, int], num_extracted: int, letter: Optional[str]
) -> Optional[float]:
    if num_extracted == 0 or letter is None:
        return None
    return counts.get(letter, 0) / num_extracted


# ---------------------------------------------------------------------
# Loading segmentation masks for a dataset index
# ---------------------------------------------------------------------
def load_masks_for_index(patch_dir: str, idx: int) -> Optional[np.ndarray]:
    base = f"nb_{idx}"
    mask_path = os.path.join(patch_dir, f"{base}_masks.npz")
    if not os.path.exists(mask_path):
        return None
    data = np.load(mask_path)
    masks = data["masks"]  # (N, H, W) bool
    return masks


# ---------------------------------------------------------------------
# Segmentation-based patch search
# ---------------------------------------------------------------------
def find_segmentation_patches(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    masks: np.ndarray,  # (N, H, W) bool
    num_patches: int,
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    min_sc_drop: float,
) -> Dict[str, Any]:

    original_image = data_dict["image"]
    w, h = original_image.size

    baseline_res = self_consistency_single(
        llm, processor, data_dict, num_samples_sc, temperature, top_p
    )
    baseline_num = baseline_res["num_extracted"]
    baseline_counts = baseline_res["letter_counts"]
    baseline_majority_letter = baseline_res["majority_letter"]
    gold_letter = data_dict["gold_letter"]

    baseline_sc_orig = sc_wrt_letter(
        baseline_counts, baseline_num, baseline_majority_letter
    )
    baseline_sc_new = baseline_sc_orig
    baseline_sc_gold = baseline_res["gold_sc"]

    print(f"Baseline majority letter: {baseline_majority_letter}")
    print(f"Baseline SC (orig/new answer): {baseline_sc_orig}")
    print(f"Baseline SC (gold): {baseline_sc_gold}")

    if masks.dtype != bool:
        masks = masks.astype(bool)
    N, H, W = masks.shape
    assert (H, W) == (h, w), "Mask size must match image size."

    global_mask = np.zeros((H, W), dtype=bool)
    selected_indices: List[int] = []

    final_sc_orig = baseline_sc_orig
    final_sc_new = baseline_sc_new
    final_sc_gold = baseline_sc_gold
    final_majority_letter = baseline_majority_letter

    for i in range(num_patches):
        print(f"\n--- Searching for segmentation patch {i+1}/{num_patches} ---")

        current_dp = data_dict.copy()
        current_dp["image"] = apply_binary_mask_to_image(original_image, global_mask)
        cur_res = self_consistency_single(
            llm, processor, current_dp, num_samples_sc, temperature, top_p
        )
        cur_num = cur_res["num_extracted"]
        cur_counts = cur_res["letter_counts"]
        cur_majority_letter = cur_res["majority_letter"]

        orig_letter = baseline_majority_letter
        new_letter = cur_majority_letter

        cur_sc_orig = sc_wrt_letter(cur_counts, cur_num, orig_letter)
        cur_sc_new = sc_wrt_letter(cur_counts, cur_num, new_letter)
        cur_sc_gold = sc_wrt_letter(cur_counts, cur_num, gold_letter)

        print(f"Current majority letter: {cur_majority_letter}")
        print(f"Current SC wrt original answer: {cur_sc_orig}")
        print(f"Current SC wrt new answer: {cur_sc_new}")
        print(f"Current SC wrt gold: {cur_sc_gold}")

        if cur_sc_orig is None:
            print("No valid SC baseline (no parsed answers); stopping.")
            break

        best_idx = None

        best_sc_after = None
        best_drop = 0.0

        # Collect all unused masks and build candidate datapoints
        candidate_indices: List[int] = []
        candidate_dps: List[Dict[str, Any]] = []

        for m_idx in range(N):
            if m_idx in selected_indices:
                continue
            candidate_indices.append(m_idx)

            candidate_mask = global_mask | masks[m_idx]
            cand_dp = data_dict.copy()
            cand_dp["image"] = apply_binary_mask_to_image(original_image, candidate_mask)
            candidate_dps.append(cand_dp)

        if not candidate_dps:
            print("No remaining candidate masks; stopping.")
            break

        # Batched SC for all candidates
        batch_results = self_consistency_batch(
            llm,
            processor,
            candidate_dps,
            num_samples_sc=num_samples_sc,
            temperature=temperature,
            top_p=top_p,
        )

        # Evaluate SC drop for each candidate
        for m_idx, cand_res in zip(candidate_indices, batch_results):
            cand_num = cand_res["num_extracted"]
            cand_counts = cand_res["letter_counts"]

            cand_sc_orig = sc_wrt_letter(cand_counts, cand_num, orig_letter)
            if cand_sc_orig is None:
                drop = 0.0
            else:
                drop = cur_sc_orig - cand_sc_orig

            print(
                f"  Mask {m_idx}: SC_orig_after={cand_sc_orig}, "
                f"drop={drop:.4f}"
            )

            if cand_sc_orig is not None and drop > best_drop and drop >= min_sc_drop:
                best_drop = drop
                best_idx = m_idx
                best_sc_after = cand_sc_orig


        if best_idx is None:
            print(
                f"No segmentation patch produced SC drop >= {min_sc_drop}; "
                "stopping search for this image."
            )
            break

        print(
            f"Selected mask index {best_idx} with SC_orig_after={best_sc_after}, "
            f"drop={best_drop:.4f}"
        )
        global_mask |= masks[best_idx]
        selected_indices.append(best_idx)

    final_dp = data_dict.copy()
    final_dp["image"] = apply_binary_mask_to_image(original_image, global_mask)
    final_res = self_consistency_single(
        llm, processor, final_dp, num_samples_sc, temperature, top_p
    )
    final_num = final_res["num_extracted"]
    final_counts = final_res["letter_counts"]
    final_majority_letter = final_res["majority_letter"]

    final_sc_orig = sc_wrt_letter(final_counts, final_num, baseline_majority_letter)
    final_sc_new = sc_wrt_letter(final_counts, final_num, final_majority_letter)
    final_sc_gold = sc_wrt_letter(final_counts, final_num, gold_letter)

    print("\n=== Final metrics after all selected patches ===")
    print(f"Final majority letter: {final_majority_letter}")
    print(f"Final SC wrt original answer: {final_sc_orig}")
    print(f"Final SC wrt new answer: {final_sc_new}")
    print(f"Final SC wrt gold: {final_sc_gold}")

    return {
        "baseline_majority_letter": baseline_majority_letter,
        "baseline_sc_orig": baseline_sc_orig,
        "baseline_sc_new": baseline_sc_new,
        "baseline_sc_gold": baseline_sc_gold,
        "final_majority_letter": final_majority_letter,
        "final_sc_orig": final_sc_orig,
        "final_sc_new": final_sc_new,
        "final_sc_gold": final_sc_gold,
        "selected_mask_indices": selected_indices,
        "global_mask": global_mask,
        "baseline_num_extracted": baseline_num,
        "final_num_extracted": final_num,
    }


# ---------------------------------------------------------------------
# Visualization & saving (fixed layout)
# ---------------------------------------------------------------------
def create_seg_comparison_visualization(
    original_image: Image.Image,
    masked_image: Image.Image,
    question: str,
    gold_answer: str,
    answer_choices: List[str],
    baseline_majority_letter: Optional[str],
    final_majority_letter: Optional[str],
    baseline_sc_orig: Optional[float],
    baseline_sc_new: Optional[float],   # still passed, just not shown
    baseline_sc_gold: Optional[float],
    final_sc_orig: Optional[float],
    final_sc_new: Optional[float],
    final_sc_gold: Optional[float],
    output_path: str,
):
    """
    Side-by-side original vs masked, with SC scores:

    Pre-mask (original image):
      - SC wrt original answer
      - SC wrt gold answer

    Post-mask:
      - SC wrt original answer
      - SC wrt new answer
      - SC wrt gold answer

    Question & gold answer at the top of the text block under the images.
    Font size scales with image size.
    """
    width, height = original_image.size

    # Text block below images
    text_block_h = int(0.35 * height) + 80
    canvas_w = width * 2
    canvas_h = height + text_block_h

    combined = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Paste images at the top (no vertical shift)
    combined.paste(original_image, (0, 0))
    combined.paste(masked_image, (width, 0))

    draw = ImageDraw.Draw(combined)

    # Font size ~ proportional to width
    base_font_size = max(14, min(28, width // 25))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", base_font_size
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            max(12, base_font_size - 2),
        )
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Helper to trim text
    def trim(text: str, max_len: int = 120) -> str:
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    # Map letter -> text
    def letter_to_text(letter: Optional[str]) -> str:
        if letter is None:
            return "N/A"
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(answer_choices):
            return answer_choices[idx]
        return "N/A"

    baseline_answer_text = letter_to_text(baseline_majority_letter)
    final_answer_text = letter_to_text(final_majority_letter)

    # All text starts below the images
    y = height + 10

    # Question and gold answer (span left side)
    draw.text((10, y), "Question:", fill="black", font=font)
    y += base_font_size + 4
    draw.text((10, y), trim(question), fill="black", font=font_small)
    y += base_font_size + 8
    draw.text((10, y), f"Gold Answer: {gold_answer}", fill="black", font=font_small)

    # Leave a gap, then left/right stats
    y_stats = y + base_font_size + 10

    left_x = 10
    right_x = width + 10

    # Pre-mask stats (ORIGINAL IMAGE)
    # Now only:
    #  - SC wrt original answer
    #  - SC wrt gold answer
    pre_lines = [
        "Pre-mask (original image)",
        f"Model majority answer: {baseline_answer_text}",
        f"SC wrt original answer: {baseline_sc_orig if baseline_sc_orig is not None else 'N/A'}",
        f"SC wrt gold answer: {baseline_sc_gold if baseline_sc_gold is not None else 'N/A'}",
    ]
    cur_y = y_stats
    for line in pre_lines:
        draw.text((left_x, cur_y), line, fill="black", font=font_small)
        cur_y += base_font_size + 2

    # Post-mask stats (on right column) – unchanged
    post_lines = [
        "Post-mask (with selected patches)",
        f"Model majority answer: {final_answer_text}",
        f"SC wrt original answer: {final_sc_orig if final_sc_orig is not None else 'N/A'}",
        f"SC wrt new answer: {final_sc_new if final_sc_new is not None else 'N/A'}",
        f"SC wrt gold answer: {final_sc_gold if final_sc_gold is not None else 'N/A'}",
    ]
    cur_y = y_stats
    for line in post_lines:
        draw.text((right_x, cur_y), line, fill="black", font=font_small)
        cur_y += base_font_size + 2

    combined.save(output_path)
    print(f"Visualization saved to {output_path}")



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print("Initializing vLLM engine...")
    llm = init_vllm_engine(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    print("Loading NaturalBench (lmms-eval) dataset...")
    ds = load_naturalbench_dataset(split=args.split)
    n_ds = len(ds)
    print(f"Dataset size: {n_ds}")

    # Resolve end_idx
    start_idx = max(0, args.start_idx)
    end_idx = n_ds if args.end_idx < 0 else min(args.end_idx, n_ds)
    if start_idx >= end_idx:
        print(f"Empty index range: start_idx={start_idx}, end_idx={end_idx}")
        return
    print(f"Processing dataset indices in range [{start_idx}, {end_idx})")

    # Find all mask files and infer their indices
    mask_files = [
        f for f in os.listdir(args.patch_dir) if f.endswith("_masks.npz")
    ]
    idxs = []
    for f in mask_files:
        m = re.match(r"nb_(\d+)_masks\.npz", f)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(idxs))

    # Restrict to [start_idx, end_idx)
    idxs = [i for i in idxs if start_idx <= i < end_idx]
    print(f"Found masks for {len(idxs)} dataset indices in the specified range.")

    for idx in idxs:
        if idx < 0 or idx >= n_ds:
            print(f"Index {idx} out of dataset range; skipping.")
            continue

        print("\n" + "=" * 80)
        print(f"Processing dataset index {idx}")
        print("=" * 80 + "\n")

        masks = load_masks_for_index(args.patch_dir, idx)
        if masks is None or masks.shape[0] == 0:
            print("No masks for this index; skipping.")
            continue

        dp = datapoint_to_dict(ds[idx])

        results = find_segmentation_patches(
            llm=llm,
            processor=processor,
            data_dict=dp,
            masks=masks,
            num_patches=args.num_patches,
            num_samples_sc=args.num_samples_sc,
            temperature=args.temperature,
            top_p=args.top_p,
            min_sc_drop=args.min_sc_drop,
        )

        original_image = dp["image"]
        masked_image = apply_binary_mask_to_image(
            original_image, results["global_mask"]
        )

        viz_path = os.path.join(args.output_dir, f"seg_comparison_{idx}.png")
        create_seg_comparison_visualization(
            original_image=original_image,
            masked_image=masked_image,
            question=dp["question"],
            gold_answer=dp["gold_answer"],
            answer_choices=dp["answer_choices"],
            baseline_majority_letter=results["baseline_majority_letter"],
            final_majority_letter=results["final_majority_letter"],
            baseline_sc_orig=results["baseline_sc_orig"],
            baseline_sc_new=results["baseline_sc_new"],
            baseline_sc_gold=results["baseline_sc_gold"],
            final_sc_orig=results["final_sc_orig"],
            final_sc_new=results["final_sc_new"],
            final_sc_gold=results["final_sc_gold"],
            output_path=viz_path,
        )

        json_path = os.path.join(args.output_dir, f"seg_results_{idx}.json")
        out_json = {
            "image_idx": idx,
            "baseline_majority_letter": results["baseline_majority_letter"],
            "final_majority_letter": results["final_majority_letter"],
            "baseline_sc_orig": results["baseline_sc_orig"],
            "baseline_sc_new": results["baseline_sc_new"],
            "baseline_sc_gold": results["baseline_sc_gold"],
            "final_sc_orig": results["final_sc_orig"],
            "final_sc_new": results["final_sc_new"],
            "final_sc_gold": results["final_sc_gold"],
            "selected_mask_indices": results["selected_mask_indices"],
            "baseline_num_extracted": results["baseline_num_extracted"],
            "final_num_extracted": results["final_num_extracted"],
            "question": dp["question"],
            "gold_answer": dp["gold_answer"],
            "gold_letter": dp["gold_letter"],
            "answer_choices": dp["answer_choices"],
            "spurious_attribute": dp["spurious_attribute"],
        }
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"Results saved to {json_path}")

    print("\n" + "=" * 80)
    print("Completed segmentation-based patch search for all indices in range")
    print("=" * 80)


if __name__ == "__main__":
    main()
