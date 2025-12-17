#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

def _early_parse_devices(argv=None) -> Tuple[int, int]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vlm_device", type=int, required=True)
    p.add_argument("--inpaint_device", type=int, required=True)
    args, _ = p.parse_known_args(argv)
    return args.vlm_device, args.inpaint_device


_vlm_dev, _inp_dev = _early_parse_devices()

if _vlm_dev == _inp_dev:
    raise SystemExit(
        f"ERROR: --vlm_device and --inpaint_device must be different GPUs. "
        f"Got both = {_vlm_dev}. (Run them on separate GPUs to avoid allocator conflicts.)"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = f"{_vlm_dev},{_inp_dev}"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import datasets
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from diffusers import StableDiffusionInpaintPipeline


def parse_args():
    parser = argparse.ArgumentParser()

    # General stuff
    parser.add_argument("--patch_dir", type=str, required=True)
    parser.add_argument("--output_dir",type=str, default="inpainting_search_result",)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--vlm_device", type=int, required=True)
    parser.add_argument("--inpaint_device", type=int, required=True)

    # Search Params
    parser.add_argument("--num_patches", type=int, default=1)
    parser.add_argument("--num_samples_sc", type=int, default=15)
    parser.add_argument("--min_sc_drop", type=float, default=0.0)

    # vLLM settings
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-32B-Instruct-FP8")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

    # Inpainting params
    parser.add_argument("--inpaint_model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--inpaint_prompt",type=str,default="Fill the masked region naturally, removing whatever objects appear there. Preserve the rest of the image exactly.")
    parser.add_argument("--inpaint_negative_prompt", type=str,default="text, watermark, logo, artifacts, distortion, blur")
    parser.add_argument("--inpaint_steps", type=int, default=30)
    parser.add_argument("--inpaint_guidance", type=float, default=7.5)
    parser.add_argument("--inpaint_strength", type=float, default=1.0)
    parser.add_argument("--inpaint_batch_size", type=int, default=8)

    parser.add_argument("--use_qwen_diffusion_prompt", action="store_true",help="Qwen generates diffusion prompt instead of default prompt",)
    parser.add_argument("--qwen_prompt_temperature",type=float,default=0.2)
    parser.add_argument("--qwen_prompt_top_p", type=float, default=0.9,help="Top-p for the Qwen->diffusion-prompt generation call.",)
    parser.add_argument("--qwen_prompt_max_tokens",type=int,default=512,help="Max tokens for the Qwen->diffusion-prompt generation call.")

    # Optional fixed resolution casting
    parser.add_argument("--image_size",type=int,default=768)
    parser.add_argument("--no_resize",action="store_true")

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    return parser.parse_args()


def load_naturalbench_dataset(split: str = "test"):
    return datasets.load_dataset("BaiqiL/NaturalBench-lmms-eval", split=split)

def datapoint_to_dict(dp):
    img = dp["Image"]
    question_raw = dp["Question"]
    q_type = dp["Question_Type"]
    ans_raw = str(dp["Answer"]).strip()

    base_question = question_raw
    answer_choices: List[str] = []
    gold_letter: Optional[str] = None
    gold_answer: Optional[str] = None

    if q_type == "yes_no":
        answer_choices = ["Yes", "No"]
        if ans_raw.lower().startswith("y"):
            gold_letter = "A"
            gold_answer = "Yes"
        else:
            gold_letter = "B"
            gold_answer = "No"
    else:
        optA, optB = "Option A", "Option B"

        # "Option: A:...; B:...;"
        if "Option:" in question_raw:
            head, tail = question_raw.split("Option:", 1)
            base_question = head.strip()

            segs = [s.strip() for s in tail.split(";") if s.strip()]
            parsed = {}
            for s in segs:
                if s.startswith("A:"):
                    parsed["A"] = s[2:].strip()
                elif s.startswith("B:"):
                    parsed["B"] = s[2:].strip()

            if parsed.get("A"):
                optA = parsed["A"]
            if parsed.get("B"):
                optB = parsed["B"]

        else:
            # "A. ... B. ..."
            m2 = re.search(r"\bA\.\s*(.*?)\s*B\.\s*(.*)$", question_raw)
            if m2:
                optA = m2.group(1).strip()
                optB = m2.group(2).strip()
                base_question = question_raw.split("A.")[0].strip()

        answer_choices = [optA, optB]
        gold_letter = ans_raw.upper()
        idx = ord(gold_letter) - ord("A")
        gold_answer = answer_choices[idx] if 0 <= idx < len(answer_choices) else ans_raw

    return {
        "image": img,
        "question": base_question,
        "answer_choices": answer_choices,
        "gold_answer": gold_answer,
        "gold_letter": gold_letter,
        "spurious_attribute": q_type,
    }



def resize_pad_to_square(img: Image.Image, size: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_rs = img.resize((new_w, new_h), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (240, 240, 240))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(img_rs, (pad_left, pad_top))
    return canvas, (pad_left, pad_top, new_w, new_h)


def resize_pad_mask(mask_bool: np.ndarray, orig_size: Tuple[int, int], size: int, pad_info: Tuple[int, int, int, int]) -> np.ndarray:
    orig_w, orig_h = orig_size
    if mask_bool.shape != (orig_h, orig_w):
        raise ValueError(f"Mask shape {mask_bool.shape} does not match orig image (H,W)=({orig_h},{orig_w}).")

    pad_left, pad_top, new_w, new_h = pad_info

    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    mask_rs = mask_img.resize((new_w, new_h), resample=Image.NEAREST)

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(mask_rs, (pad_left, pad_top))
    return (np.array(canvas) > 0)


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
    prompt += "\nPlease provide the final answer in this JSON format: {'final_answer_letter': <letter>}\n"
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

# LOADING VLM AND SELF CONSISTENCY
def init_vllm_engine(model_id: str, tensor_parallel_size: int, gpu_memory_utilization: float, max_model_len: int, seed: int) -> LLM:
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


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def sc_wrt_letter(counts: Dict[str, int], num_extracted: int, letter: Optional[str]) -> Optional[float]:
    if num_extracted == 0 or letter is None:
        return None
    return counts.get(letter, 0) / num_extracted


def self_consistency_batch(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    images: List[Image.Image],
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    question = make_text_prompt(data_dict["question"], data_dict["answer_choices"])

    vllm_inputs = []
    for im in images:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": im}, {"type": "text", "text": question}],
        }]
        vllm_inputs.append(prepare_inputs_for_vllm(messages, processor))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=num_samples_sc,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    results = []
    gold_letter = data_dict["gold_letter"]
    for out in outputs:
        num_extracted = 0
        letter_counts: Dict[str, int] = {}
        for gen in out.outputs:
            parsed = parse_qwen_output(gen.text)
            if parsed is not None:
                num_extracted += 1
                letter_counts[parsed] = letter_counts.get(parsed, 0) + 1

        majority_letter = max(letter_counts.items(), key=lambda x: x[1])[0] if letter_counts else None
        gold_sc = (letter_counts.get(gold_letter, 0) / num_extracted) if (num_extracted > 0 and gold_letter) else None

        results.append({
            "num_extracted": num_extracted,
            "letter_counts": letter_counts,
            "majority_letter": majority_letter,
            "gold_sc": gold_sc,
        })

    return results


# INPAINTING
def load_inpaint_pipe(model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"):
    """
    Match your working test pattern, but force device to cuda:1 (remapped inpaint GPU).
    """
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    return pipe, device


def mask_bool_to_pil_L(mask_bool: np.ndarray) -> Image.Image:
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def inpaint_batch(
    pipe: StableDiffusionInpaintPipeline,
    device: str,
    base_image: Image.Image,
    masks_bool: List[np.ndarray],
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    strength: float,
    batch_size: int,
    seed: int,
) -> List[Image.Image]:
    out_images: List[Image.Image] = []
    if len(masks_bool) == 0:
        return out_images

    use_cuda = device.startswith("cuda")
    gen = torch.Generator(device=device).manual_seed(seed) if use_cuda else None

    for s in range(0, len(masks_bool), batch_size):
        chunk = masks_bool[s:s + batch_size]
        imgs = [base_image] * len(chunk)
        masks = [mask_bool_to_pil_L(m) for m in chunk]

        with torch.inference_mode():
            w, h = base_image.size
            h8 = (h // 8) * 8
            w8 = (w // 8) * 8

            res = pipe(
                prompt=[prompt] * len(chunk),
                negative_prompt=[negative_prompt] * len(chunk),
                image=imgs,
                mask_image=masks,
                height=h8,
                width=w8,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                strength=strength,
                generator=gen,
            )

        out_images.extend(res.images)

        if use_cuda:
            torch.cuda.synchronize(torch.device(device))

    return out_images


def apply_black_mask(img: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    """
    Return an image where masked pixels are set to black
    """
    arr = np.array(img).copy()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr[mask_bool] = 0
    return Image.fromarray(arr)


def _letter_to_choice_text(letter: Optional[str], answer_choices: List[str]) -> str:
    if letter is None:
        return "N/A"
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(answer_choices):
        return answer_choices[idx]
    return "N/A"


def qwen_diffusion_prompts_batch(
    llm: LLM,
    processor: AutoProcessor,
    data_dict: Dict[str, Any],
    masked_images: List[Image.Image],
    baseline_letter: str,
    baseline_answer_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[Dict[str, str]]:
    """
    For each masked image, ask Qwen: what should go in the blacked-out region to make baseline answer less likely?
    Returns list of dicts: {"inpaint_prompt": str, "negative_prompt": str}
    """
    q = data_dict["question"]
    choices = data_dict["answer_choices"]

    instruction = (
        "You are helping generate an image-edit instruction for an inpainting model.\n\n"
        "Task:\n"
        f"Given the VQA question and answer choices, the baseline answer is {baseline_letter} ({baseline_answer_text}).\n"
        f"Propose what to place INSIDE the blacked-out region to make {baseline_letter} less likely.\n\n"
        "Constraints:\n"
        "- Only describe what should appear INSIDE the region.\n"
        "- Keep it <= 20 words, concrete, visual.\n"
        "- Preserve everything outside the region.\n\n"
        "Return JSON ONLY:\n"
        "{\"inpaint_prompt\": \"...\", \"negative_prompt\": \"...\"}\n\n"
        f"Question: {q}\n"
        f"Answer choices: {choices}\n"
    )

    vllm_inputs = []
    for im in masked_images:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": im}, {"type": "text", "text": instruction}],
        }]
        vllm_inputs.append(prepare_inputs_for_vllm(messages, processor))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=1,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)


    out: List[Dict[str, str]] = []
    for o in outputs:
        raw = o.outputs[0].text if o.outputs else ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0).replace("'", '"'))
                ip = str(obj.get("inpaint_prompt", "")).strip()
                nprompt = str(obj.get("negative_prompt", "")).strip()
                if ip:
                    out.append({"inpaint_prompt": ip, "negative_prompt": nprompt})
                    continue
            except Exception:
                pass

        out.append({
            "inpaint_prompt": "A plausible object that contradicts the baseline answer.",
            "negative_prompt": "blurry, watermark, distorted, artifacts, text",
        })

    print(out)
    return out

# SEARCH
def load_masks_for_index(patch_dir: str, idx: int) -> Optional[np.ndarray]:
    mask_path = os.path.join(patch_dir, f"nb_{idx}_masks.npz")
    if not os.path.exists(mask_path):
        return None
    data = np.load(mask_path)
    return data["masks"]  # (N, H, W) bool


def find_segmentation_patches_inpaint(
    llm: LLM,
    processor: AutoProcessor,
    inpaint_pipe: StableDiffusionInpaintPipeline,
    inpaint_device: str,
    data_dict: Dict[str, Any],
    masks: np.ndarray,  # (N,H,W) bool
    num_patches: int,
    num_samples_sc: int,
    temperature: float,
    top_p: float,
    min_sc_drop: float,
    inpaint_prompt: str,
    inpaint_negative_prompt: str,
    inpaint_steps: int,
    inpaint_guidance: float,
    inpaint_strength: float,
    inpaint_batch_size: int,
    seed: int,
    use_qwen_diffusion_prompt: bool,
    qwen_prompt_temperature: float,
    qwen_prompt_top_p: float,
    qwen_prompt_max_tokens: int,
) -> Dict[str, Any]:

    original_image = data_dict["image"]
    w0, h0 = original_image.size

    baseline_res = self_consistency_batch(
        llm, processor, data_dict, [original_image], num_samples_sc, temperature, top_p
    )[0]
    baseline_num = baseline_res["num_extracted"]
    baseline_counts = baseline_res["letter_counts"]
    baseline_majority_letter = baseline_res["majority_letter"]
    gold_letter = data_dict["gold_letter"]

    baseline_sc_orig = sc_wrt_letter(baseline_counts, baseline_num, baseline_majority_letter)
    baseline_sc_new = baseline_sc_orig
    baseline_sc_gold = baseline_res["gold_sc"]

    print(f"Baseline majority letter: {baseline_majority_letter}")
    print(f"Baseline SC (orig/new answer): {baseline_sc_orig}")
    print(f"Baseline SC (gold): {baseline_sc_gold}")

    if masks.dtype != bool:
        masks = masks.astype(bool)

    N, H, W = masks.shape
    assert (H, W) == (h0, w0), "Mask size must match original image size."

    global_mask = np.zeros((H, W), dtype=bool)
    selected_indices: List[int] = []
    qwen_prompts_used: List[Dict[str, Any]] = []

    baseline_answer_text = _letter_to_choice_text(baseline_majority_letter, data_dict["answer_choices"])

    for step_i in range(num_patches):
        print(f"\n--- Searching for segmentation patch {step_i + 1}/{num_patches} ---")

        if global_mask.any():
            if use_qwen_diffusion_prompt and (baseline_majority_letter is not None):
                masked_for_qwen = apply_black_mask(original_image, global_mask)
                pobj = qwen_diffusion_prompts_batch(
                    llm=llm,
                    processor=processor,
                    data_dict=data_dict,
                    masked_images=[masked_for_qwen],
                    baseline_letter=baseline_majority_letter,
                    baseline_answer_text=baseline_answer_text,
                    temperature=qwen_prompt_temperature,
                    top_p=qwen_prompt_top_p,
                    max_tokens=qwen_prompt_max_tokens,
                )[0]

                cur_prompt = pobj.get("inpaint_prompt", "") or inpaint_prompt
                cur_neg = pobj.get("negative_prompt", "") or inpaint_negative_prompt
            else:
                cur_prompt = inpaint_prompt
                cur_neg = inpaint_negative_prompt

            cur_img = inpaint_batch(
                pipe=inpaint_pipe,
                device=inpaint_device,
                base_image=original_image,
                masks_bool=[global_mask],
                prompt=cur_prompt,
                negative_prompt=cur_neg,
                num_steps=inpaint_steps,
                guidance=inpaint_guidance,
                strength=inpaint_strength,
                batch_size=1,
                seed=seed + 1000 + step_i,
            )[0]
        else:
            cur_img = original_image

        cur_res = self_consistency_batch(
            llm, processor, data_dict, [cur_img], num_samples_sc, temperature, top_p
        )[0]
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

        cand_indices = [m_idx for m_idx in range(N) if m_idx not in selected_indices]
        cand_masks = [global_mask | masks[m_idx] for m_idx in cand_indices]


        if use_qwen_diffusion_prompt and (orig_letter is not None):
            cand_masked_imgs = [apply_black_mask(original_image, cm) for cm in cand_masks]

            prompt_objs = qwen_diffusion_prompts_batch(
                llm=llm,
                processor=processor,
                data_dict=data_dict,
                masked_images=cand_masked_imgs,
                baseline_letter=orig_letter,
                baseline_answer_text=baseline_answer_text,
                temperature=qwen_prompt_temperature,
                top_p=qwen_prompt_top_p,
                max_tokens=qwen_prompt_max_tokens,
            )
            print(f"Prompt for SD {prompt_objs}")

            cand_images: List[Image.Image] = []
            for cm, pobj in zip(cand_masks, prompt_objs):
                p = pobj.get("inpaint_prompt", "") or inpaint_prompt
                nprompt = pobj.get("negative_prompt", "") or inpaint_negative_prompt
                img_out = inpaint_batch(
                    pipe=inpaint_pipe,
                    device=inpaint_device,
                    base_image=original_image,
                    masks_bool=[cm],
                    prompt=p,
                    negative_prompt=nprompt,
                    num_steps=inpaint_steps,
                    guidance=inpaint_guidance,
                    strength=inpaint_strength,
                    batch_size=1,
                    seed=seed + 2000 + step_i,
                )[0]
                cand_images.append(img_out)

            qwen_prompts_used.append({
                "step": step_i,
                "baseline_letter": orig_letter,
                "baseline_answer_text": baseline_answer_text,
                "candidates": [
                    {"mask_idx": mi, "inpaint_prompt": pobj.get("inpaint_prompt", ""),
                     "negative_prompt": pobj.get("negative_prompt", "")}
                    for mi, pobj in zip(cand_indices, prompt_objs)
                ],
            })
        else:
            cand_images = inpaint_batch(
                pipe=inpaint_pipe,
                device=inpaint_device,
                base_image=original_image,
                masks_bool=cand_masks,
                prompt=inpaint_prompt,
                negative_prompt=inpaint_negative_prompt,
                num_steps=inpaint_steps,
                guidance=inpaint_guidance,
                strength=inpaint_strength,
                batch_size=inpaint_batch_size,
                seed=seed + 2000 + step_i,
            )


        cand_sc_results = self_consistency_batch(
            llm, processor, data_dict, cand_images, num_samples_sc, temperature, top_p
        )

        best_idx = None
        best_drop = 0.0
        best_sc_after = None

        for m_idx, res in zip(cand_indices, cand_sc_results):
            cand_num = res["num_extracted"]
            cand_counts = res["letter_counts"]
            cand_sc_orig = sc_wrt_letter(cand_counts, cand_num, orig_letter)
            drop = (cur_sc_orig - cand_sc_orig) if (cand_sc_orig is not None) else 0.0

            print(f"  Mask {m_idx}: SC_orig_after={cand_sc_orig}, drop={drop:.4f}")

            if cand_sc_orig is not None and drop > best_drop and drop >= min_sc_drop:
                best_drop = drop
                best_idx = m_idx
                best_sc_after = cand_sc_orig

        if best_idx is None:
            print(f"No patch produced SC drop >= {min_sc_drop}; stopping search for this image.")
            break

        print(f"Selected mask index {best_idx} with SC_orig_after={best_sc_after}, drop={best_drop:.4f}")
        global_mask |= masks[best_idx]
        selected_indices.append(best_idx)

    if global_mask.any():
        final_img = inpaint_batch(
            pipe=inpaint_pipe,
            device=inpaint_device,
            base_image=original_image,
            masks_bool=[global_mask],
            prompt=inpaint_prompt,
            negative_prompt=inpaint_negative_prompt,
            num_steps=inpaint_steps,
            guidance=inpaint_guidance,
            strength=inpaint_strength,
            batch_size=1,
            seed=seed + 9999,
        )[0]
    else:
        final_img = original_image

    final_res = self_consistency_batch(
        llm, processor, data_dict, [final_img], num_samples_sc, temperature, top_p
    )[0]
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
        "final_image": final_img,
        "qwen_prompts_used": qwen_prompts_used,
    }


# VISUALIZATION
def create_comparison_visualization(
    original_image: Image.Image,
    perturbed_image: Image.Image,
    question: str,
    gold_answer: str,
    answer_choices: List[str],
    baseline_majority_letter: Optional[str],
    final_majority_letter: Optional[str],
    baseline_sc_orig: Optional[float],
    baseline_sc_new: Optional[float],
    baseline_sc_gold: Optional[float],
    final_sc_orig: Optional[float],
    final_sc_new: Optional[float],
    final_sc_gold: Optional[float],
    output_path: str,
):
    import textwrap

    def _load_font(sz: int):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
        except Exception:
            return ImageFont.load_default()

    def _letter_to_text(letter: Optional[str]) -> str:
        if letter is None:
            return "N/A"
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(answer_choices):
            return answer_choices[idx]
        return "N/A"

    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "N/A"
        return f"{x:.4f}"

    def _resize_to_fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = im.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        if (nw, nh) == (w, h):
            return im
        return im.resize((nw, nh), resample=Image.BICUBIC)

    def _measure_multiline(draw, lines, font, line_gap=4):
        heights = []
        max_w = 0
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_w = max(max_w, w)
            heights.append(h)
        total_h = sum(heights) + max(0, len(lines) - 1) * line_gap
        return max_w, total_h


    margin = 24
    gutter = 24
    header_gap = 10
    line_gap = 4
    MAX_COL_W = 820
    MAX_IMG_H = 520

    # Fonts
    font_title = _load_font(22)
    font_body = _load_font(16)
    font_small = _load_font(14)

    # Resize images to fit the same box
    orig_rs = _resize_to_fit(original_image, MAX_COL_W, MAX_IMG_H)
    pert_rs = _resize_to_fit(perturbed_image, MAX_COL_W, MAX_IMG_H)

    col_w = max(orig_rs.size[0], pert_rs.size[0], 500)

    baseline_answer_text = _letter_to_text(baseline_majority_letter)
    final_answer_text = _letter_to_text(final_majority_letter)

    left_caption = [
        "Pre (original image)",
        f"Model majority answer: {baseline_answer_text}",
        f"SC wrt original answer: {_fmt(baseline_sc_orig)}",
        f"SC wrt new answer: {_fmt(baseline_sc_new)}",
        f"SC wrt gold answer: {_fmt(baseline_sc_gold)}",
    ]
    right_caption = [
        "Post (inpainted with selected patches)",
        f"Model majority answer: {final_answer_text}",
        f"SC wrt original answer: {_fmt(final_sc_orig)}",
        f"SC wrt new answer: {_fmt(final_sc_new)}",
        f"SC wrt gold answer: {_fmt(final_sc_gold)}",
    ]


    header_lines = ["Question:"]
    header_lines += textwrap.wrap(question, width=90) if question else ["(missing)"]
    header_lines.append(f"Gold Answer: {gold_answer}")

    scratch = Image.new("RGB", (10, 10), "white")
    d0 = ImageDraw.Draw(scratch)

    _, header_h = _measure_multiline(d0, header_lines, font_body, line_gap=line_gap)
    _, left_cap_h = _measure_multiline(d0, left_caption, font_small, line_gap=line_gap)
    _, right_cap_h = _measure_multiline(d0, right_caption, font_small, line_gap=line_gap)
    cap_h = max(left_cap_h, right_cap_h)

    canvas_w = margin * 2 + col_w * 2 + gutter
    canvas_h = (
        margin
        + header_h
        + header_gap
        + max(orig_rs.size[1], pert_rs.size[1])
        + header_gap
        + cap_h
        + margin
    )

    combined = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(combined)

    x = margin
    y = margin
    draw.text((x, y), header_lines[0], fill="black", font=font_title)
    y += (draw.textbbox((0, 0), header_lines[0], font=font_title)[3] - draw.textbbox((0, 0), header_lines[0], font=font_title)[1]) + 6

    for ln in header_lines[1:]:
        draw.text((x, y), ln, fill="black", font=font_body)
        bbox = draw.textbbox((0, 0), ln, font=font_body)
        y += (bbox[3] - bbox[1]) + line_gap

    y += header_gap

    left_x = margin
    right_x = margin + col_w + gutter
    img_y = y

    ox = left_x + (col_w - orig_rs.size[0]) // 2
    px = right_x + (col_w - pert_rs.size[0]) // 2
    combined.paste(orig_rs, (ox, img_y))
    combined.paste(pert_rs, (px, img_y))

    y = img_y + max(orig_rs.size[1], pert_rs.size[1]) + header_gap

    def _draw_caption(col_x: int, lines: List[str]):
        yy = y
        for ln in lines:
            draw.text((col_x, yy), ln, fill="black", font=font_small)
            bbox = draw.textbbox((0, 0), ln, font=font_small)
            yy += (bbox[3] - bbox[1]) + line_gap

    _draw_caption(left_x, left_caption)
    _draw_caption(right_x, right_caption)

    combined.save(output_path)
    print(f"Visualization saved to {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch sees {torch.cuda.device_count()} GPUs (expected 2).")
    print(f"vLLM GPU (cuda:0) name: {torch.cuda.get_device_name(0)}")
    print(f"inpaint GPU (cuda:1) name: {torch.cuda.get_device_name(1)}")

    print(f"Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print("Initializing vLLM engine (will use cuda:0)...")
    llm = init_vllm_engine(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    print("Initializing inpainting pipeline (will use cuda:1)...")
    inpaint_pipe, inpaint_device = load_inpaint_pipe(args.inpaint_model_id)
    print(f"Inpaint device: {inpaint_device}")

    print("Loading NaturalBench dataset...")
    ds = load_naturalbench_dataset(split=args.split)
    n_ds = len(ds)
    print(f"Dataset size: {n_ds}")

    start_idx = max(0, args.start_idx)
    end_idx = (n_ds - 1) if args.end_idx < 0 else min(args.end_idx, n_ds - 1)

    if start_idx > end_idx:
        raise SystemExit(f"ERROR: start_idx ({start_idx}) > end_idx ({end_idx}).")

    print(f"Processing dataset indices in range: [{start_idx}, {end_idx}] (inclusive)")

    mask_files = [f for f in os.listdir(args.patch_dir) if f.endswith("_masks.npz")]
    idxs = []
    for f in mask_files:
        m = re.match(r"nb_(\d+)_masks\.npz", f)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(idxs))
    print(f"Found masks for {len(idxs)} dataset indices.")

    for idx in idxs:
        if idx < 0 or idx >= n_ds:
            print(f"Index {idx} out of dataset range; skipping.")
            continue
        
        if idx < start_idx or idx > end_idx:
            continue

        print("\n" + "=" * 80)
        print(f"Processing dataset index {idx}")
        print("=" * 80 + "\n")

        masks = load_masks_for_index(args.patch_dir, idx)
        if masks is None or masks.shape[0] == 0:
            print("No masks for this index; skipping.")
            continue

        dp = datapoint_to_dict(ds[idx])

        if not args.no_resize:
            orig_img = dp["image"]
            orig_w, orig_h = orig_img.size
            img_sq, pad_info = resize_pad_to_square(orig_img, args.image_size)
            dp["image"] = img_sq

            resized_masks = []
            for m in masks:
                resized_masks.append(resize_pad_mask(m, (orig_w, orig_h), args.image_size, pad_info))
            masks = np.stack(resized_masks, axis=0).astype(bool)
            print(f"Resized/padded to {args.image_size}x{args.image_size}. Masks now: {masks.shape}")

        results = find_segmentation_patches_inpaint(
            llm=llm,
            processor=processor,
            inpaint_pipe=inpaint_pipe,
            inpaint_device=inpaint_device,
            data_dict=dp,
            masks=masks,
            num_patches=args.num_patches,
            num_samples_sc=args.num_samples_sc,
            temperature=args.temperature,
            top_p=args.top_p,
            min_sc_drop=args.min_sc_drop,
            inpaint_prompt=args.inpaint_prompt,
            inpaint_negative_prompt=args.inpaint_negative_prompt,
            inpaint_steps=args.inpaint_steps,
            inpaint_guidance=args.inpaint_guidance,
            inpaint_strength=args.inpaint_strength,
            inpaint_batch_size=args.inpaint_batch_size,
            seed=args.seed,
            use_qwen_diffusion_prompt=args.use_qwen_diffusion_prompt,
            qwen_prompt_temperature=args.qwen_prompt_temperature,
            qwen_prompt_top_p=args.qwen_prompt_top_p,
            qwen_prompt_max_tokens=args.qwen_prompt_max_tokens,
        )

        original_image = dp["image"]
        final_image = results["final_image"]

        viz_path = os.path.join(args.output_dir, f"inpaint_comparison_{idx}.png")
        create_comparison_visualization(
            original_image=original_image,
            perturbed_image=final_image,
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

        json_path = os.path.join(args.output_dir, f"inpaint_results_{idx}.json")
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
            "inpaint_model_id": args.inpaint_model_id,
            "inpaint_prompt": args.inpaint_prompt,
            "inpaint_negative_prompt": args.inpaint_negative_prompt,
            "inpaint_steps": args.inpaint_steps,
            "inpaint_guidance": args.inpaint_guidance,
            "inpaint_strength": args.inpaint_strength,
            "use_qwen_diffusion_prompt": args.use_qwen_diffusion_prompt,
            "qwen_prompt_temperature": args.qwen_prompt_temperature,
            "qwen_prompt_top_p": args.qwen_prompt_top_p,
            "qwen_prompt_max_tokens": args.qwen_prompt_max_tokens,
            "qwen_prompts_used": results.get("qwen_prompts_used", []),
            "image_size": None if args.no_resize else args.image_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vlm_device_arg": args.vlm_device,
            "inpaint_device_arg": args.inpaint_device,
        }
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"Results saved to {json_path}")

    print("\n" + "=" * 80)
    print("Completed inpainting-based segmentation patch search for all indices with masks")
    print("=" * 80)


if __name__ == "__main__":
    main()
