import json
import random
from collections import Counter

view = 'view'
data_study_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/view_statistics_{view}_study.json'
data_view_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/view_statistics_{view}_view.json'
with open(data_study_path, 'r') as f:
    data_study = json.load(f)
with open(data_view_path, 'r') as f:
    data_view = json.load(f)

labels = list(data_view.keys())
studies = list(data_study.keys())

# split ratios (adjust if desired)
train_frac, val_frac, test_frac = 0.8, 0.1, 0.1
random.seed(42)

# sort labels by how many studies they appear in (ascending)
labels_sorted = sorted(labels, key=lambda l: len(data_view[l]))

study_split = {}

for label in labels_sorted:
    studies_with_label = list(data_view[label].keys())
    # remove studies already assigned by earlier (rarer) labels
    remaining = [s for s in studies_with_label if s not in study_split]
    if not remaining:
        continue

    random.shuffle(remaining)
    n = len(remaining)

    # initial counts by fractions
    n_train = min(int(n * train_frac), n - 2)  # leave room for val/test
    n_val = max(int(n * val_frac), 1)
    n_test = n - n_train - n_val

    # assign splits for this label's remaining studies
    start = 0
    for s in remaining[start:start + n_train]:
        study_split[s] = "train"
    start += n_train
    for s in remaining[start:start + n_val]:
        study_split[s] = "val"
    start += n_val
    for s in remaining[start:]:
        study_split[s] = "test"

# assign any studies that never appeared in data_view (if any) to train
for s in studies:
    if s not in study_split:
        study_split[s] = "train"

# study_split is the desired output: mapping study -> split
print("split counts:", Counter(study_split.values()))

label_sets = {"train": set(), "val": set(), "test": set()}
for study, study_split_label in study_split.items():
    label_sets[study_split_label].update(data_study[study].keys())
labels_counts = {split: len(labels) for split, labels in label_sets.items()}
print("label counts per split:", labels_counts)

output_path = f'/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_view/split_{view}.json'
with open(output_path, "w") as f:
    json.dump(study_split, f, indent=2)