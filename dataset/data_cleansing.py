import csv
import os
import numpy as np
from collections import Counter

data_path = '/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/'

labels = {
    'LHF': ['0', '1', '0.0', '1.0'],
    'RHF': ['0', '1', '0.0', '1.0'],
    'DF': ['0', '1', '0.0', '1.0'],
    'LAD': ['0', '1', '0.0', '1.0'],
    'LVD': ['0', '1', '0.0', '1.0'],
    'RAD': ['0', '1', '0.0', '1.0'],
    'RVD': ['0', '1', '0.0', '1.0'],
    'AVA': ['0', '1', '0.0', '1.0'],
    'MVA': ['0', '1', '0.0', '1.0'],
    'TVA': ['0', '1', '0.0', '1.0'],
    'PVA': ['0', '1', '0.0', '1.0'],
    'PE': ['0', '1', '0.0', '1.0'],
    'LVH': ['0', '1', '0.0', '1.0'],
    'IMT': ['0', '1', '0.0', '1.0'],
    'IS': ['0', '1', '0.0', '1.0'],
    'LHF_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RHF_severity': ['normal', 'mild', 'moderate', 'severe'],
    'DF_severity': ['normal', 'grade 1', 'grade 2', 'grade 3'],
    'LAD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'LVD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RAD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RVD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'AVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    'MVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    'TVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    'PVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    'PE_severity': ['small', 'moderate', 'large', 'tamponade physiology'],
    'LVH_severity': ['normal', 'mild', 'moderate', 'severe'],
}

def process_dataset_csv(dataset_csv_path):
    with open(dataset_csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def process_csv(path, task):
    """
    read csv file, return a (label, path) set in which all the paths exists without repetition, and all labels are correct
    """
    data = set()
    pathset = set()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # row = [label, split, path]
            assert len(row) == 3, f"a row with length {len(row)} is found"
            if os.path.exists(data_path + row[2]):
                row[0] = row[0].lower()
                if row[0] not in labels[task]:
                    if '0' in labels[task]:
                        continue
                    for i in range(len(labels[task]) - 1, 0, -1):
                        if labels[task][i] in row[0]:
                            row[0] = labels[task][i]
                            break
                    if row[0] not in labels[task]:
                        continue
                elif row[0] == '0.0':
                    row[0] = '0'
                elif row[0] == '1.0':
                    row[0] = '1'
                data.add((row[0], row[2]))
                pathset.add(row[2])
        # assert len(data) == len(pathset), f"len(data) is {len(data)} while len(pathset) is {len(pathset)}"
    return data

def remove_repetition(data):
    """
    remove repeated path in the data
    """
    path_counts = Counter(path for _, path in data)
    nonrepetitive_paths = {path for path, count in path_counts.items() if count == 1}
    nonrepetitive_tuples = [[item[0], item[1]] for item in data if item[1] in nonrepetitive_paths]
    return nonrepetitive_tuples

def split(data_dir):
    """
    split the data into train, val, test set with ratio 48:1:1
    """
    n = data_dir.shape[0]
    num_train = int(n * 0.96)
    num_val = int(n * 0.02)
    num_test = n - num_train - num_val

    data_split = np.concatenate([
        np.repeat('train', num_train),
        np.repeat('val', num_val),
        np.repeat('test', num_test)
    ])
    data_split = np.random.permutation(data_split)

    data_dir_split = np.concatenate([
        data_dir[:, 0, np.newaxis],
        data_split[:, np.newaxis],
        data_dir[:, 1, np.newaxis]
    ], axis=1)

    sorted_indices = np.argsort(data_dir_split[:, 2])
    data_dir_split = data_dir_split[sorted_indices]
    return data_dir_split

def add_file(data):
    """
    add file in the path
    """
    data_file = []
    cnt = 0
    # error_cnt = 0
    for row in data:
        label, split, path = row

        long_path = data_path + path
        # if not os.path.exists(long_path):
        #     error_cnt += 1
        #     continue
        
        # the path should exist
        for file in os.listdir(long_path):
            data_file.append([label, split, path + '/' + file])

        cnt += 1
        if cnt % 1000 == 0:
            print(f"{cnt} rows have been processed")

    # print(f"error_cnt is {error_cnt}")

    return data_file

def output(data, path):
    with open(path, 'w') as f:
        f.write('label,split,path\n')
        for row in data:
            f.write(','.join(row) + '\n')

def process(task_set):
    dataset_csv_dir = '/home/patxiao/ECHO/label_v2'
    # dataset_output_dir = '/home/patxiao/ECHO/label_dataset_v2'
    dataset_output_dir = 'dataset_csv/ECHO/label_dataset_v2'
    for task in task_set:
        print(f"processing {task}😊😊😊")

        dataset_csv_path = os.path.join(dataset_csv_dir, task + '.csv')
        data_dir_set = process_csv(dataset_csv_path, task)
        data_dir = np.array(remove_repetition(data_dir_set), dtype=object)

        data_dir_split = list(split(data_dir))
        print(data_dir_split[:10])
        data_dir_split_path = os.path.join(dataset_output_dir, 'dir_' + task + '.csv')
        output(data_dir_split, data_dir_split_path)

        data_file_split = add_file(data_dir_split)
        print(data_file_split[:10])
        data_file_split_path = os.path.join(dataset_output_dir, task + '.csv')
        output(data_file_split, data_file_split_path)

def check(task_set):
    # dataset_csv_dir = 'dataset_csv/ECHO/label_dataset_v2'
    dataset_csv_dir = '/home/patxiao/ECHO/label_dataset_v1'
    stats = {}
    labels_ = {}
    for task in task_set:
        stats_task = {"train": {}, "val": {}, "test": {}}
        print(f"checking {task}😊😊😊")
        dataset_csv_path = os.path.join(dataset_csv_dir, task + '.csv')
        labels_[task] = set()
        with open(dataset_csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'label':
                    continue
                labels_[task].add(row[0])
                if row[0] not in stats_task[row[1]]:
                    stats_task[row[1]][row[0]] = 0
                stats_task[row[1]][row[0]] += 1
        stats[task] = len(labels_[task])
        print(labels_[task])
        print(stats_task)
    print(stats)
    # print(labels)


# task_set = ['AVA', 'DF', 'IMT', 'IS', 'LAD', 'LHF', 'LVD', 'LVH', 'MVA', 'PAP', 'PE', 'PVA', 'RAD', 'RAP', 'RHF', 'RVD', 'TVA', 'AVA_severity', 'DF_severity', 'IMT', 'IS', 'LAD_severity', 'LHF_severity', 'LVD_severity', 'LVH_severity', 'MVA_severity', 'PAP', 'PE_severity', 'PVA_severity', 'RAD_severity', 'RAP', 'RHF_severity', 'RVD_severity', 'TVA_severity']
task_set = ['AVA', 'DF', 'IMT', 'IS', 'LAD', 'LHF', 'LVD', 'LVH', 'MVA', 'PE', 'PVA', 'RAD', 'RHF', 'RVD', 'TVA', 'AVA_severity', 'DF_severity', 'IMT', 'IS', 'LAD_severity', 'LHF_severity', 'LVD_severity', 'LVH_severity', 'MVA_severity', 'PE_severity', 'PVA_severity', 'RAD_severity', 'RHF_severity', 'RVD_severity', 'TVA_severity']
# task_set = ['PE_severity', 'PVA_severity', 'RAD_severity', 'RHF_severity', 'RVD_severity', 'TVA_severity']
task_set = ['HF', 'HF_mini']
# process(task_set)
check(task_set)