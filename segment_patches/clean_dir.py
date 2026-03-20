#!/usr/bin/env python3

import os
import json
import argparse
import re
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove results with no selected patches "
                    "(empty selected_mask_indices)."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing seg_results_*.json and seg_comparison_*.png",
    )
    return parser.parse_args()


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON {path}: {e}")
        return None


def main():
    args = parse_args()
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"[ERROR] {results_dir} is not a directory")
        return

    # Find all seg_results_*.json files
    files = os.listdir(results_dir)
    json_files = [f for f in files if f.startswith("seg_results_") and f.endswith(".json")]

    print(f"Found {len(json_files)} JSON result files in {results_dir}")

    pattern = re.compile(r"seg_results_(\d+)\.json")

    removed_count = 0

    for json_name in json_files:
        m = pattern.match(json_name)
        if not m:
            continue

        idx = m.group(1)
        json_path = os.path.join(results_dir, json_name)
        data = load_json(json_path)
        if data is None:
            continue

        selected = data.get("selected_mask_indices", None)

        # Treat missing field or empty list as "no patch found"
        if not selected:
            png_name = f"seg_comparison_{idx}.png"
            png_path = os.path.join(results_dir, png_name)

            print(f"[INFO] No patches for index {idx} (selected_mask_indices={selected}).")
            print(f"       Deleting {json_path}")
            try:
                os.remove(json_path)
            except FileNotFoundError:
                print(f"       JSON file already missing: {json_path}")
            except Exception as e:
                print(f"       [WARN] Failed to delete JSON {json_path}: {e}")

            if os.path.exists(png_path):
                print(f"       Deleting {png_path}")
                try:
                    os.remove(png_path)
                except Exception as e:
                    print(f"       [WARN] Failed to delete PNG {png_path}: {e}")
            else:
                print(f"       PNG not found (skipping): {png_path}")

            removed_count += 1

    print(f"\nDone. Removed {removed_count} pairs with empty selected_mask_indices.")


if __name__ == "__main__":
    main()
