import csv
import os
import numpy as np
from collections import Counter
import re

data_path = '/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/'

range = {
    'PAP': [0.0, 100.0],
    'RAP': [0.0, 50.0]
}

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
            if row[0] == 'label':
                continue
            if os.path.exists(data_path + row[2]):
                label = re.findall(r'\d+\.\d+|\d+', row[0])
                assert len(label) > 0, f"label {row[0]} is not a number"
                if len(label) > 1:
                    print(f"label {row[0]} has more than one number")
                label = [float(i) for i in label if float(i) >= range[task][0]]
                if len(label) == 0:
                    print(f"label {row[0]} is not in the range {range[task]}")
                    continue
                label = sum(label) / len(label)
                data.add((str(label), row[2]))
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
    split the data into train, val, test set with ratio 7:1:2
    """
    n = data_dir.shape[0]
    num_train = int(n * 0.7)
    num_val = int(n * 0.1)
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
    dataset_csv_dir = '/mnt/hanoverdev/data/patxiao/ECHO/processed_label_v4'
    # dataset_output_dir = '/home/patxiao/ECHO/label_dataset_v2'
    dataset_output_dir = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4'
    for task in task_set:
        print(f"processing {task}😊😊😊")

        dataset_csv_path = os.path.join(dataset_csv_dir, task + '.csv')
        data_dir_set = process_csv(dataset_csv_path, task)
        data_dir = np.array(remove_repetition(data_dir_set), dtype=object)

        data_dir_split = list(split(data_dir))
        print(data_dir_split[:10])
        data_dir_split_path = os.path.join(dataset_output_dir, task + '.csv')
        output(data_dir_split, data_dir_split_path)

def check(task_set):
    # dataset_csv_dir = 'dataset_csv/ECHO/label_dataset_v2'
    dataset_csv_dir = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_mini'
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


task_set = ['PAP', 'RAP']
process(task_set)
# check(task_set)