import json
import random

studies_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_label.json"
output_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_split.json"

with open(studies_path, 'r') as f:
    studies = json.load(f)
    
# Split the studies into train, val, and test sets with ratios 18:1:1
num_train = int(len(studies) * 0.9)
num_val = int(len(studies) * 0.05)
num_test = len(studies) - num_train - num_val

splits = ["train"] * num_train + ["val"] * num_val + ["test"] * num_test
random.shuffle(splits)

studies_split = {study: split for study, split in zip(studies, splits)}
# Save the split studies to a new JSON file
with open(output_path, 'w') as f:
    json.dump(studies_split, f, indent=4)
print(f"Split studies saved to {output_path}")