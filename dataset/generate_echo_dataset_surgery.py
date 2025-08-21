import pandas as pd
import json
import os
from tqdm import tqdm
import random
import pickle

split_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_split.json"
label_path = "/mnt/hanoverdev/data/patxiao/surgery_indication_v4.pkl"
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_study_only/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(label_path, 'rb') as f:
    data = pickle.load(f)
with open(split_path, 'r') as f:
    splits = json.load(f)

new_df = {"label":[], "split":[], "path": []}
not_in_split_num = 0
for study in data:
    data[study]["surgery_indication_estimated"] = int(data[study]["surgery_indication_estimated"])
    if data[study]["surgery_indication_estimated"] == -1:
        continue
    assert data[study]["surgery_indication_estimated"] in [0, 1], f"Invalid label {data[study]['surgery_indication_estimated']} for study {study}"

    # assert path in studies, f"Path {path} not in studies"
    if study not in splits:
        continue
        splits[study] = "train" if random.random() < 0.8 else "test"
        not_in_split_num += 1
    new_df["label"].append(data[study]["surgery_indication_estimated"])
    new_df["split"].append(splits[study])
    new_df["path"].append(study)

new_df = pd.DataFrame(new_df)
output_path = os.path.join(output_dir, "surgery_indication.csv")
new_df.to_csv(output_path, index=False)
print(f"Filtered dataset saved to {output_path}")
if not_in_split_num > 0:
    print(f"Number of samples not in split: {not_in_split_num}")