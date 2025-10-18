import pandas as pd
import json

# view = 'window'
view = 'view'

file_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/{view}.csv'
df = pd.read_csv(file_path)

data_study = {}
data_view = {}
for row in df.itertuples():
    label = row.label
    study = row.path.split('/')[0]
    if study not in data_study:
        data_study[study] = {}
    if label not in data_study[study]:
        data_study[study][label] = 0
    data_study[study][label] += 1

    if label not in data_view:
        data_view[label] = {}
    if study not in data_view[label]:
        data_view[label][study] = 0
    data_view[label][study] += 1

# print(data_study)
# print(data_view)

output_study_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/view_statistics_{view}_study.json'
output_view_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/view_statistics_{view}_view.json'
with open(output_study_path, 'w') as f:
    json.dump(data_study, f, indent=4)
with open(output_view_path, 'w') as f:
    json.dump(data_view, f, indent=4)