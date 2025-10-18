import pandas as pd
import numpy as np
from tqdm import tqdm

view_data_path = '/mnt/hanoverdev/data/patxiao/ECHO_view_labels/ViewLabels20250720.csv'
view_data = pd.read_csv(view_data_path)

output_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/window.csv'

n = len(view_data)
num_train = int(n * 0.8)
num_val = int(n * 0.1)
num_test = n - num_train - num_val

data_split = np.concatenate([
    np.repeat('train', num_train),
    np.repeat('val', num_val),
    np.repeat('test', num_test)
])
data_split = np.random.permutation(data_split)

new_df = {"label": [], "split": [], "path": []}
for i in tqdm(range(n)):
    if not view_data['is_video'].iloc[i]:
        continue
    if view_data['window'].iloc[i] == "Right Flank" or view_data['window'].iloc[i] == "GLS analysis":
        continue
    if view_data['window'].iloc[i] is np.nan:
        if view_data['view'].iloc[i] == "3D ECHO":
            new_df["label"].append("3D ECHO")
        else:
            raise ValueError(f"Unexpected view type: ({view_data['window'].iloc[i]}, {view_data['view'].iloc[i]})")
    else:
        new_df["label"].append(view_data["window"].iloc[i])
    new_df["split"].append(data_split[i])
    new_df["path"].append(f"{view_data['case_id'].iloc[i]}/{view_data['file_name'].iloc[i]}.pt")

new_df = pd.DataFrame(new_df)
# new_df.to_csv(output_path, index=False)

# print(f"Output saved to {output_path}")
print(f"Classes: {new_df['label'].unique()}")

output_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/view.csv'

n = len(view_data)
num_train = int(n * 0.8)
num_val = int(n * 0.1)
num_test = n - num_train - num_val

data_split = np.concatenate([
    np.repeat('train', num_train),
    np.repeat('val', num_val),
    np.repeat('test', num_test)
])
data_split = np.random.permutation(data_split)

new_df = {"label": [], "split": [], "path": []}
for i in tqdm(range(n)):
    if not view_data['is_video'].iloc[i]:
        continue
    if view_data['view'].iloc[i] == 'IAS':
        continue
    if view_data['view'].iloc[i] is np.nan:
        if view_data['window'].iloc[i] == "3D ECHO":
            new_df["label"].append("3D ECHO")
        else:
            # print(f"Unexpected view type: ({view_data['window'].iloc[i]}, {view_data['view'].iloc[i]})")
            continue
    else:
        new_df["label"].append(view_data["view"].iloc[i])
    new_df["split"].append(data_split[i])
    new_df["path"].append(f"{view_data['case_id'].iloc[i]}/{view_data['file_name'].iloc[i]}.pt")

new_df = pd.DataFrame(new_df)
# new_df.to_csv(output_path, index=False)

# print(f"Output saved to {output_path}")
print(f"Classes: {new_df['label'].unique()}")