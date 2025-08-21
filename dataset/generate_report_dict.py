import pandas as pd
import json
import os
from tqdm import tqdm

study_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_report.json"
raw_data_path = "/mnt/hanoverdev/data/hanwen/ECHO/deidentified/"
output_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_reports.json"

with open(study_path, 'r') as f:
    studies = json.load(f)

data = dict()
for study in tqdm(studies, desc=f"Processing studies"):
    report_dir = os.path.join(raw_data_path, study)
    if not os.path.exists(report_dir):
        print(f"Report directory {report_dir} does not exist")
        continue
    reports = [f for f in os.listdir(report_dir) if f.endswith('.txt')]
    assert len(reports) >= 1, f"Study {study} has no report files"
    report_file = sorted(reports)[-1]
    with open(os.path.join(report_dir, report_file), 'r') as f:
        report = f.read()
    data[study] = report

with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)
