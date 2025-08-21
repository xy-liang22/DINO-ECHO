import os
import pandas as pd
import json

csv_dir = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/'

task_set = ['AV_regurgitation', 'AV_stenosis', 'LAD', 'LHF', 'LVD', 'LVH', 'MV_regurgitation', 'MV_stenosis', 'PE', 'PV_regurgitation', 'PV_stenosis', 'RAD', 'RHF', 'RVD', 'TV_regurgitation', 'TV_stenosis']

template = {
    "in task_severity not in task": [],
    "in task not in task_severity": [],
    "negative in task positive in task_severity": [],
    "positive in task negative in task_severity": [],
}

results = {}

for task in task_set:
    # results[task] = template.copy()
    results[task] = {
        "in task_severity not in task": [],
        "in task not in task_severity": [],
        "negative in task positive in task_severity": [],
        "positive in task negative in task_severity": [],
    }
    assert os.path.exists(os.path.join(csv_dir, task + '.csv')), f"File {task}.csv does not exist in {csv_dir}"
    assert os.path.exists(os.path.join(csv_dir, task + '_severity.csv')), f"File {task}_severity.csv does not exist in {csv_dir}"
    print(f"✊✊✊ Processing {task}... ✊✊✊")
    
    task_csv = pd.read_csv(os.path.join(csv_dir, task + '.csv'))
    task_severity_csv = pd.read_csv(os.path.join(csv_dir, task + '_severity.csv'))
    task_dict = {path: label for path, label in zip(task_csv['path'], task_csv['label'])}
    task_severity_dict = {path: label for path, label in zip(task_severity_csv['path'], task_severity_csv['label'])}
    path_set = set(task_dict.keys()).union(set(task_severity_dict.keys()))
    print(len(path_set))
    
    cnt = 10
    for path in path_set:
        assert path in task_dict or path in task_severity_dict, f"Path {path} not found in either task or task_severity"
        
        if path in task_severity_dict and path not in task_dict:
            results[task]["in task_severity not in task"].append(path + ', ' + task_severity_dict[path])
        elif path in task_dict and task_dict[path] == 0 and path not in task_severity_dict:
            results[task]["in task not in task_severity"].append(path + ', ' + str(task_dict[path]))
        elif path in task_dict and path in task_severity_dict:
            if task_dict[path] == 0 and task_severity_dict[path] != 'normal':
                results[task]["negative in task positive in task_severity"].append(path + ', ' + str(task_dict[path]) + ', ' + task_severity_dict[path])
            elif task_dict[path] != 0 and task_severity_dict[path] == 'normal':
                results[task]["positive in task negative in task_severity"].append(path + ', ' + str(task_dict[path]) + ', ' + task_severity_dict[path])
                # if cnt:
                #     print(path, task_dict[path], task_dict[path] == '0', task_severity_dict[path], task_severity_dict[path] == 'normal')
                #     cnt -= 1
    print(len(results[task]["in task_severity not in task"]))
    print(len(results[task]["in task not in task_severity"]))
    print(len(results[task]["negative in task positive in task_severity"]))
    print(len(results[task]["positive in task negative in task_severity"]))
    print("✊✊✊ Processing completed. ✊✊✊")
            
save_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/label_check.json"

with open(save_path, 'w') as f:
    json.dump(results, f, indent=4)
    print(f"🎉🎉🎉 Results saved successfully to {save_path}. 🎉🎉🎉")


# normal
# exists in ${task}_severity but no in ${task}
# exists and is negative in ${task} but not in ${task}_severity
# positive in ${task}_severity but negative in ${task}
# negative in ${task}_severity but positive in ${task}