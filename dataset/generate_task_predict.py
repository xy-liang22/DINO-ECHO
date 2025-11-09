import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm
import torch

study_labels_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_labels.json"
with open(study_labels_path, 'r') as f:
    label_dict = json.load(f)

# mode = "full" # "full" or "exist"
mode = "exist"
output_path = f"/data/ECHO/llava_data_label/label_predict_{mode}.csv"

def get_predict_file(task):
    predict_path = f"/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_task_predict/{task}_predict/predictions.csv"
    with open(predict_path, 'r') as f:
        df = pd.read_csv(f)
    return df

tasks = ["LHF", "RHF", "DF", "LAD", "RAD", "LVD", "RVD", "AV_regurgitation", "AV_stenosis", "MV_regurgitation", "MV_stenosis", "TV_regurgitation", "TV_stenosis", "PV_stenosis", "PV_regurgitation", "PE", "LVH"]
predict_list = {task: get_predict_file(task) for task in tasks}

data = pd.DataFrame()

for i, study in enumerate(label_dict.keys()):
    row = {'study': study}
    for task in tasks:
        df = predict_list[task]
        pred = df['prediction'][i]
        if mode == "exist" and label_dict[study][task] is None:
            row[task] = None
        else:
            row[task] = df['prediction'][i]
    data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)

data.columns = ['study', *tasks]
data.to_csv(output_path, index=False)

