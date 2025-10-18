import json
import pandas as pd
import os
from tqdm import tqdm

study_label_list_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_label.json'
study_report_list_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_report.json'

view_data_path = '/mnt/hanoverdev/data/patxiao/ECHO_view_labels/ViewLabels20250720.csv'

with open(study_label_list_path, 'r') as f:
    study_label_list = json.load(f)
with open(study_report_list_path, 'r') as f:
    study_report_list = json.load(f)

view_data_pd = pd.read_csv(view_data_path)

view_data_study = set(view_data_pd['case_id'].to_list())

print("View study cases length:", len(view_data_study))
print("View study in study_label length:", len(view_data_study & set(study_label_list)))
print("View study in study_report length:", len(view_data_study & set(study_report_list)))

echo_embedding_dir = '/data/ECHO/dinov2_study_original1_embeddings'
sum = 0
total_videos = 0
for i in tqdm(range(len(view_data_pd))):
    case_id = view_data_pd.iloc[i]['case_id']
    file_name = view_data_pd.iloc[i]['file_name']
    pt_path = f'{echo_embedding_dir}/{case_id}/{file_name}.pt'
    if os.path.exists(pt_path):
        sum += 1
    if view_data_pd.iloc[i]['is_video']:
        total_videos += 1

print("Existing pt files:", sum, "Total pt video files:", total_videos, "Total files", len(view_data_pd))