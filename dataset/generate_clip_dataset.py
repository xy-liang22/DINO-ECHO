import pandas as pd
import json
import os
from tqdm import tqdm

study_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_report.json"
split_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_split.json"
best_video_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/best_video.json"
raw_data_path = "/mnt/hanoverdev/data/hanwen/ECHO/deidentified/"
npy_dir = "/mnt/hanoverdev/data/patxiao/ECHO_numpy/original_size"
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/"

with open(study_path, 'r') as f:
    studies = json.load(f)
with open(split_path, 'r') as f:
    splits = json.load(f)
with open(best_video_path, 'r') as f:
    best_video = json.load(f)

for split in ["train", "val", "test"]:
    studies_cnt_cplit = [study for study, split_ in splits.items() if split_ == split and study in studies]
    print(f"Number of {split} studies: {len(studies_cnt_cplit)}")
    data = {"video_path":list(), "report":list()}
    for study in tqdm(studies_cnt_cplit, desc=f"Processing {split} studies"):
        if study not in studies:
            print(f"Study {study} not found in studies")
            continue
        if study not in best_video:
            print(f"Study {study} not found in best_video")
            continue
        report_dir = os.path.join(raw_data_path, study)
        if not os.path.exists(report_dir):
            print(f"Report directory {report_dir} does not exist")
            continue
        reports = [f for f in os.listdir(report_dir) if f.endswith('.txt')]
        assert len(reports) >= 1, f"Study {study} has no report files"
        report_file = sorted(reports)[-1]
        with open(os.path.join(report_dir, report_file), 'r') as f:
            report = f.read()
            
        data["video_path"].append(os.path.join(npy_dir, best_video[study]))
        data["report"].append(report)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(output_dir, f"{split}_report.csv"), index=False)