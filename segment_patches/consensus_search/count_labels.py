import os
import json

target_dir = "consensus_patches"

count = 0
for file in os.listdir(target_dir):
    if file.endswith(".json"):
        count += 1
        json_path = os.path.join(target_dir, file)
        with open(json_path, "r") as f:
            data = json.load(f)
            if data.get("selected_mask_indices", None) is not None:
                count += 1
print(f"Found {count} labels")