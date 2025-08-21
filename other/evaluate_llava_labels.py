import pandas as pd
import torch
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score, recall_score)

groundtruth_label_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4"
llava_label_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4_by_llava"
result_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/LLaVA_label_results/results.csv"

task_set = ['AV_regurgitation_severity', 'AV_regurgitation', 'AV_stenosis_severity', 'AV_stenosis', 'AV_vegetations', 'DF_severity', 'DF', 'IMT', 'IS', 'LAD_severity', 'LAD', 'LHF_severity', 'LHF', 'LVD_severity', 'LVD', 'LVH_severity', 'LVH', 'MV_regurgitation_severity', 'MV_regurgitation', 'MV_stenosis_severity', 'MV_stenosis', 'MV_vegetations', 'PE_severity', 'PE', 'PV_regurgitation_severity', 'PV_regurgitation', 'PV_stenosis_severity', 'PV_stenosis', 'PV_vegetations', 'RAD_severity', 'RAD', 'RHF_severity', 'RHF', 'RVD_severity', 'RVD', 'TV_regurgitation_severity', 'TV_regurgitation', 'TV_stenosis_severity', 'TV_stenosis', 'TV_vegetations']

results_df = {"task": [], "accuracy": [], "bacc": [], "total_num": [], "right_num": [], "recall": [], "precision": [], "f1": []}

for task in task_set:
    print(f"😊😊😊 Processing task {task}")
    groundtruth_df = pd.read_csv(f"{groundtruth_label_dir}/{task}.csv")
    llava_df = pd.read_csv(f"{llava_label_dir}/{task}.csv")

    groundtruch_dict = dict()
    llava_dict = dict()
    for idx in range(len(groundtruth_df)):
        groundtruch_dict[groundtruth_df['path'][idx]] = groundtruth_df['label'][idx]
    for idx in range(len(llava_df)):
        llava_dict[llava_df['path'][idx]] = llava_df['label'][idx]

    studies = list(set(groundtruth_df['path'].tolist()) & set(llava_df['path'].tolist()))
    print(f"Task: {task}, Number of common studies: {len(studies)}")

    groundtruch_labels = [groundtruch_dict[study] for study in studies]
    llava_labels = [llava_dict[study] for study in studies]
    labels = sorted(list(set(groundtruch_labels + llava_labels)))
    groundtruch_labels = [labels.index(label) for label in groundtruch_labels]
    llava_labels = [labels.index(label) for label in llava_labels]

    assert len(labels) <= 4, f"Number of labels {len(labels)} exceeds 4 for task {task}. Please check the data."
    assert len(labels) <= 2 or "severity" in task, f"Task {task} has more than 2 labels but does not contain 'severity'. Please check the data."

    # compute accuracy, bacc
    accuracy = sum(1 for gt, ll in zip(groundtruch_labels, llava_labels) if gt == ll) / len(studies)
    groundtruch_labels = torch.tensor(groundtruch_labels)
    llava_labels = torch.tensor(llava_labels)
    bacc = balanced_accuracy_score(groundtruch_labels, llava_labels)
    results_df["task"].append(task)
    results_df["accuracy"].append(accuracy)
    results_df["bacc"].append(bacc)
    results_df["total_num"].append(len(studies))
    results_df["right_num"].append(sum(1 for gt, ll in zip(groundtruch_labels, llava_labels) if gt == ll))
    if len(labels) == 2:
        f1 = f1_score(groundtruch_labels, llava_labels, average='binary')
        precision = precision_score(groundtruch_labels, llava_labels, average='binary')
        recall = recall_score(groundtruch_labels, llava_labels, average='binary')
        # print(f"Task: {task}, Accuracy: {accuracy:.4f}, BACC: {bacc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        results_df["f1"].append(f1)
        results_df["precision"].append(precision)
        results_df["recall"].append(recall)
    else:
        results_df["f1"].append('')
        results_df["precision"].append('')
        results_df["recall"].append('')

results_df = pd.DataFrame(results_df)
results_df.to_csv(result_path, index=False)