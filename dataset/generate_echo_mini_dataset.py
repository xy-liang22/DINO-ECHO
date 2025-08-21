import pandas as pd
import json
import os
from tqdm import tqdm
import random

study_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_label.json"
split_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_split.json"
best_video_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/best_video.json"
previous_label_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/"
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_mini_new/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(study_path, 'r') as f:
    studies = json.load(f)
with open(split_path, 'r') as f:
    splits = json.load(f)
with open(best_video_path, 'r') as f:
    best_video = json.load(f)

task_set = ['AV_regurgitation_severity', 'AV_regurgitation', 'AV_stenosis_severity', 'AV_stenosis', 'AV_vegetations', 'DF_severity', 'DF', 'IMT', 'IS', 'LAD_severity', 'LAD', 'LHF_severity', 'LHF', 'LVD_severity', 'LVD', 'LVH_severity', 'LVH', 'MV_regurgitation_severity', 'MV_regurgitation', 'MV_stenosis_severity', 'MV_stenosis', 'MV_vegetations', 'PE_severity', 'PE', 'PV_regurgitation_severity', 'PV_regurgitation', 'PV_stenosis_severity', 'PV_stenosis', 'PV_vegetations', 'RAD_severity', 'RAD', 'RHF_severity', 'RHF', 'RVD_severity', 'RVD', 'TAPSE_severity', 'TV_regurgitation_severity', 'TV_regurgitation', 'TV_stenosis_severity', 'TV_stenosis', 'TV_vegetations']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for task in task_set:
    print(f"Processing task: {task}")
    csv_path = os.path.join(previous_label_dir, f"{task}.csv")
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    # indices = df[splits[df["path"]] == "train"].index.tolist()
    indices = [i for i in range(len(df)) if splits[df["path"][i]] == "train"]
    random_indices = sorted(random.sample(indices, int(len(indices) * 0.1)))
    print(f"Number of train samples: {len(indices)}, Number of random samples: {len(random_indices)}")
    new_df = {"label":[], "split":[], "path": []}
    for i, (label, split, path) in tqdm(enumerate(zip(df["label"], df["split"], df["path"])), desc=f"Processing {task}"):
        if splits[path] == "train" and i not in random_indices:
            continue
        assert path in studies, f"Path {path} not in studies"
        new_df["label"].append(label)
        new_df["split"].append(splits[path])
        # new_df["path"].append(best_video[path])
        new_df["path"].append(path)
    new_df = pd.DataFrame(new_df)
    print(f"Number of train, val, test samples:{len(new_df[new_df['split'] == 'train'])}, {len(new_df[new_df['split'] == 'val'])}, {len(new_df[new_df['split'] == 'test'])}")
    output_path = os.path.join(output_dir, f"{task}.csv")
    new_df.to_csv(output_path, index=False)
    print(f"Filtered dataset saved to {output_path}")