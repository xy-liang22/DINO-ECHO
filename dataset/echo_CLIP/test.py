import os
import json

directory = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_labels.json"

with open(directory, 'r') as f:
    data = json.load(f)
    print(data["1.2.840.113619.2.391.2103.1526041714.1.1"])
    print(data["1.2.840.113619.2.391.2103.1526041714.1.1"]["AV_regurgitation"])
    print(data["1.2.840.113619.2.391.2103.1526041714.1.1"]["AV_regurgitation"] is None)
    negative_set = set()
    for study in data:
        no_true = True
        for task in data[study]:
            if data[study][task] not in [None, '0', 'normal']:
                no_true = False
        if no_true:
            negative_set.add(study)
    print(f"negative_set = {negative_set}")
    print(len(negative_set))