import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm

study_list_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_report.json'
with open(study_list_path, 'r') as f:
    study_list = json.load(f)

n = 10
# studies = np.random.choice(study_list, n, replace=False).tolist()
studies = study_list
studies = sorted(studies)
# print(studies)

label_dict = {'window': {'3D ECHO': 0, 'Apical': 1, 'Parasternal': 2, 'SSN Aortic arch': 3, 'Subcostal': 4}, 'view': {'2C': 0, '3C': 1, '3D ECHO': 2, '4C': 3, '5C': 4, 'Focus on Abdominal aorta': 5, 'Focus on Hepatic vein': 6, 'IVC': 7, 'PLAX': 8, 'PSAX': 9}}

embedding_dir = "/data/ECHO/dinov2_study_original1_embeddings"

split = 'test'

output_dir = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view'

for task in ['window', 'view']:
    data = {'split': list(), 'path': list(), 'video_id': list()}
    video_id = 0
    id_to_video = []
    for study in tqdm(studies, desc="Processing studies"):
        dir = f"{embedding_dir}/{study}"
        count = 0
        for file in os.listdir(dir):
            if not file.endswith('.pt'):
                continue
            file_name = f"{study}/{file}"
            data['split'].append(split)
            data['path'].append(file_name)
            data['video_id'].append(video_id)

            id_to_video.append(file_name)
            video_id += 1
            count += 1
        # print(f"{count} files in {study}")
    json_data = {'embedding_dir': embedding_dir, 'label_dict': label_dict[task], 'id_to_video': id_to_video}
    data = pd.DataFrame(data)
    data.to_csv(f"{output_dir}/{task}_predict_full.csv")
    with open(f"{output_dir}/{task}_predict_full.json", 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Finished processing {task}. Save to {output_dir}/{task}_predict_full.csv and {output_dir}/{task}_predict_full.json")

