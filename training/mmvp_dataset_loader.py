#!/usr/bin/env python3
"""
MMVP Dataset Loader for Evaluation

Loads the MMVP (Multimodal Visual Patterns) benchmark dataset.
The dataset contains 300 CLIP-blind pair image-question-answer samples designed
to evaluate vision-language models on challenging visual pattern recognition.

Dataset: https://huggingface.co/datasets/MMVP/MMVP
"""

import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from PIL import Image

# Default path to MMVP data (cloned from HuggingFace)
DEFAULT_MMVP_DIR = "/mnt/arc/mjojic/data/MMVP"


def parse_options(options_str: str) -> Tuple[List[str], List[str]]:
    """
    Parse MMVP options string like "(a) Open (b) Closed" into choices.
    
    Returns:
        Tuple of (letters, choices):
        - letters: ['a', 'b'] or ['a', 'b', 'c', 'd']
        - choices: ['Open', 'Closed']
    """
    # Match patterns like (a) or (A)
    pattern = r'\(([a-zA-Z])\)\s*([^()]+?)(?=\s*\([a-zA-Z]\)|$)'
    matches = re.findall(pattern, options_str)
    
    if not matches:
        # Fallback: try to split by common delimiters
        return [], [options_str.strip()]
    
    letters = [m[0].lower() for m in matches]
    choices = [m[1].strip() for m in matches]
    
    return letters, choices


def parse_correct_answer(answer_str: str) -> str:
    """
    Parse correct answer string like "(a)" or "(b)" to letter.
    
    Returns:
        Uppercase letter like "A" or "B"
    """
    match = re.search(r'\(([a-zA-Z])\)', answer_str)
    if match:
        return match.group(1).upper()
    # Fallback: return cleaned string
    return answer_str.strip().upper()


class MMVPDataset:
    """
    MMVP Benchmark Dataset for VLM evaluation.
    
    Each sample contains:
        - image: PIL Image (224x224)
        - question: Question text
        - options: Raw options string  
        - answer_choices: List of answer options
        - gold_letter: Correct answer letter (A, B, etc.)
        - gold_answer: Full text of correct answer
        - index: Original dataset index
    """
    
    def __init__(
        self,
        mmvp_dir: str = DEFAULT_MMVP_DIR,
        transform: Optional[callable] = None,
    ):
        """
        Initialize MMVP dataset.
        
        Args:
            mmvp_dir: Path to cloned MMVP repository
            transform: Optional transform to apply to images
        """
        self.mmvp_dir = Path(mmvp_dir)
        self.images_dir = self.mmvp_dir / "MMVP Images"
        self.questions_csv = self.mmvp_dir / "Questions.csv"
        self.transform = transform
        
        # Validate paths
        if not self.mmvp_dir.exists():
            raise FileNotFoundError(f"MMVP directory not found: {self.mmvp_dir}")
        if not self.questions_csv.exists():
            raise FileNotFoundError(f"Questions.csv not found: {self.questions_csv}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Load questions
        self.samples = self._load_questions()
        print(f"[MMVPDataset] Loaded {len(self.samples)} samples from {self.mmvp_dir}")
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load all questions from CSV."""
        samples = []
        
        with open(self.questions_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row['Index'])
                question = row['Question']
                options_str = row['Options']
                correct_answer = row['Correct Answer']
                
                # Parse options
                letters, choices = parse_options(options_str)
                
                # Parse correct answer
                gold_letter_lower = parse_correct_answer(correct_answer)
                
                # Convert letter to uppercase A, B, etc.
                if gold_letter_lower.lower() in letters:
                    gold_idx = letters.index(gold_letter_lower.lower())
                    gold_letter = chr(65 + gold_idx)  # A, B, C, D
                    gold_answer = choices[gold_idx] if gold_idx < len(choices) else ""
                else:
                    gold_letter = gold_letter_lower
                    gold_answer = correct_answer
                
                # Convert choices to A, B, C, D format
                answer_choices = choices
                
                samples.append({
                    'index': idx,
                    'question': question,
                    'options_raw': options_str,
                    'answer_choices': answer_choices,
                    'gold_letter': gold_letter,
                    'gold_answer': gold_answer,
                    'correct_answer_raw': correct_answer,
                })
        
        return samples
    
    def _load_image(self, idx: int) -> Image.Image:
        """Load image for given index."""
        image_path = self.images_dir / f"{idx}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, dataset_idx: int) -> Dict[str, Any]:
        """Get a sample by dataset index (0 to len-1)."""
        sample = self.samples[dataset_idx].copy()
        sample['image'] = self._load_image(sample['index'])
        return sample
    
    def get_by_original_index(self, original_idx: int) -> Dict[str, Any]:
        """Get a sample by its original MMVP index (1 to 300)."""
        for i, s in enumerate(self.samples):
            if s['index'] == original_idx:
                return self[i]
        raise IndexError(f"Original index {original_idx} not found")


def print_sample_info(sample: Dict[str, Any], idx: int = 0) -> None:
    """
    Print detailed information about a single MMVP sample.
    """
    print("\n" + "=" * 80)
    print(f"MMVP SAMPLE (dataset_idx={idx}, original_index={sample.get('index', 'N/A')})")
    print("=" * 80)
    
    # Print all keys
    print("\n[Keys in sample]:")
    for key in sorted(sample.keys()):
        value = sample[key]
        value_type = type(value).__name__
        if isinstance(value, Image.Image):
            print(f"  - {key}: {value_type} (size={value.size}, mode={value.mode})")
        elif isinstance(value, (list, tuple)):
            print(f"  - {key}: {value_type} (len={len(value)}) = {value}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"  - {key}: {value_type} (len={len(value)} chars)")
        else:
            print(f"  - {key}: {value_type} = {repr(value)}")
    
    print("\n[Formatted View]:")
    
    # Image info
    if "image" in sample and sample["image"] is not None:
        img = sample["image"]
        if isinstance(img, Image.Image):
            print(f"\n  Image:")
            print(f"    - Size (W x H): {img.size}")
            print(f"    - Mode: {img.mode}")
    
    # Question
    if "question" in sample:
        print(f"\n  Question: {sample['question']}")
    
    # Answer choices
    if "answer_choices" in sample:
        print(f"\n  Answer Choices:")
        for i, choice in enumerate(sample["answer_choices"]):
            letter = chr(65 + i)
            marker = " <-- CORRECT" if letter == sample.get('gold_letter') else ""
            print(f"    {letter}. {choice}{marker}")
    
    # Gold answer
    if "gold_letter" in sample:
        print(f"\n  Gold Letter: {sample['gold_letter']}")
    if "gold_answer" in sample:
        print(f"  Gold Answer: {sample['gold_answer']}")
    
    print("\n" + "=" * 80)


def explore_dataset(dataset: MMVPDataset, num_samples: int = 5) -> None:
    """
    Explore the MMVP dataset structure by printing info about multiple samples.
    """
    print("\n" + "#" * 80)
    print("# MMVP DATASET EXPLORATION")
    print("#" * 80)
    
    print(f"\nDataset info:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - MMVP directory: {dataset.mmvp_dir}")
    print(f"  - Images directory: {dataset.images_dir}")
    
    # Explore individual samples
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print_sample_info(sample, idx=i)


def get_answer_distribution(dataset: MMVPDataset) -> Dict[str, int]:
    """Get distribution of correct answers."""
    dist = {}
    for sample in dataset.samples:
        letter = sample['gold_letter']
        dist[letter] = dist.get(letter, 0) + 1
    return dist


def main():
    """
    Main function to load and explore the MMVP dataset.
    """
    print("\n" + "=" * 80)
    print("MMVP Dataset Loader - Sample Exploration")
    print("=" * 80)
    
    # Load dataset
    dataset = MMVPDataset(mmvp_dir=DEFAULT_MMVP_DIR)
    
    # Explore structure with first few samples
    explore_dataset(dataset, num_samples=5)
    
    # Print summary statistics
    print("\n" + "#" * 80)
    print("# SUMMARY STATISTICS")
    print("#" * 80)
    
    # Answer distribution
    answer_dist = get_answer_distribution(dataset)
    print(f"\nAnswer distribution:")
    for letter, count in sorted(answer_dist.items()):
        print(f"  - {letter}: {count} ({100*count/len(dataset):.1f}%)")
    
    # Image sizes (check a few)
    print(f"\nImage sizes (first 5 samples):")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        img = sample['image']
        print(f"  - Sample {i} (idx={sample['index']}): {img.size} ({img.mode})")
    
    # Question types (sample)
    print(f"\nSample questions:")
    for i in range(min(5, len(dataset))):
        q = dataset.samples[i]['question']
        print(f"  - [{i}]: {q[:80]}..." if len(q) > 80 else f"  - [{i}]: {q}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
