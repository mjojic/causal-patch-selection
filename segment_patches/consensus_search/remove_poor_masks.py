#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove consensus patches that did NOT cause a drop in self-consistency.

Only keep patches where final_sc_target < baseline_sc_target.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple


def parse_args():
    ap = argparse.ArgumentParser(
        description="Remove consensus patch files that didn't cause a drop in SC."
    )
    ap.add_argument(
        "--consensus_dir",
        type=str,
        default="consensus_patches",
        help="Directory containing consensus_*.json files",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print what would be deleted without actually deleting",
    )
    return ap.parse_args()


def get_consensus_json_files(consensus_dir: str) -> List[str]:
    """Find all consensus_*.json files in the directory."""
    p = Path(consensus_dir)
    if not p.exists():
        return []
    return sorted(p.glob("consensus_*.json"))


def should_keep_patch(json_path: str) -> Tuple[bool, dict]:
    """
    Returns (should_keep, data).
    
    should_keep=True if final_sc_target < baseline_sc_target (i.e., there was a drop).
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        baseline_sc = data.get("baseline_sc_target")
        final_sc = data.get("final_sc_target")
        
        if baseline_sc is None or final_sc is None:
            print(f"[WARN] {json_path}: missing SC values, will remove")
            return False, data
        
        # Keep only if there was a drop (final < baseline)
        return final_sc < baseline_sc, data
    
    except Exception as e:
        print(f"[ERROR] Failed to read {json_path}: {e}")
        return False, {}


def remove_files(json_path: str, data: dict, dry_run: bool):
    """Remove the consensus JSON, mask PNG, and viz PNG."""
    files_to_remove = [json_path]
    
    # Extract related file paths from the JSON
    paths = data.get("paths", {})
    
    # These paths might be relative to consensus_dir
    mask_png = paths.get("mask_png")
    viz_png = paths.get("viz_png")
    
    # Resolve relative to the directory containing the JSON
    json_dir = Path(json_path).parent
    
    if mask_png:
        mask_path = json_dir / Path(mask_png).name
        if mask_path.exists():
            files_to_remove.append(str(mask_path))
    
    if viz_png:
        viz_path = json_dir / Path(viz_png).name
        if viz_path.exists():
            files_to_remove.append(str(viz_path))
    
    # Actually remove files
    for fpath in files_to_remove:
        if dry_run:
            print(f"  [DRY RUN] Would delete: {fpath}")
        else:
            try:
                os.remove(fpath)
                print(f"  Deleted: {fpath}")
            except Exception as e:
                print(f"  [ERROR] Failed to delete {fpath}: {e}")


def main():
    args = parse_args()
    
    consensus_files = get_consensus_json_files(args.consensus_dir)
    
    if len(consensus_files) == 0:
        print(f"No consensus_*.json files found in {args.consensus_dir}")
        return
    
    print(f"Found {len(consensus_files)} consensus JSON files in {args.consensus_dir}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE: No files will be deleted ===\n")
    
    kept_count = 0
    removed_count = 0
    
    for json_path in consensus_files:
        should_keep, data = should_keep_patch(str(json_path))
        
        image_idx = data.get("image_idx", "?")
        baseline_sc = data.get("baseline_sc_target")
        final_sc = data.get("final_sc_target")
        
        if should_keep:
            # There was a drop in SC, keep it
            kept_count += 1
            print(f"[KEEP {image_idx}] baseline_sc={baseline_sc:.3f}, final_sc={final_sc:.3f} (drop={baseline_sc - final_sc:.3f})")
        else:
            # No drop (or increase), remove it
            removed_count += 1
            if baseline_sc is not None and final_sc is not None:
                delta = final_sc - baseline_sc
                print(f"[REMOVE {image_idx}] baseline_sc={baseline_sc:.3f}, final_sc={final_sc:.3f} (change={delta:+.3f})")
            else:
                print(f"[REMOVE {image_idx}] Missing SC values")
            
            remove_files(str(json_path), data, args.dry_run)
    
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Kept: {kept_count} patches (with SC drop)")
    print(f"  Removed: {removed_count} patches (no SC drop)")


if __name__ == "__main__":
    main()
