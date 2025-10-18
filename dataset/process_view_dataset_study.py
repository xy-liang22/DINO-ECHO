import pandas as pd
import json

view = 'window'
split_data_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/split_{view}.json'
dataset_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view_old/{view}.csv'

with open(split_data_path, 'r') as f:
    split_data = json.load(f)
dataset = pd.read_csv(dataset_path)

# change split according to study split
for i in range(len(dataset)):
    study = dataset['path'].iloc[i].split('/')[0]
    if study in split_data:
        dataset.at[i, 'split'] = split_data[study]
    else:
        raise ValueError(f"Study {study} not found in split data")

output_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/{view}.csv'
dataset.to_csv(output_path, index=False)
print(f"Output saved to {output_path}")