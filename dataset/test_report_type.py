import json
import os
from tqdm import tqdm

studies_dir = "/mnt/hanoverdev/data/hanwen/ECHO/original_size/"

report_types = dict()

assert os.path.exists(studies_dir), f"Directory {studies_dir} does not exist"
print("Directory exists, proceeding with processing...")


cnt_abnormal = 0
type_set = set()
# abnormal_report = []

studies_list = os.listdir(studies_dir)
print(f"Total number of studies: {len(studies_list)}")

studies_with_no_report = []
for study in tqdm(studies_list, desc="Processing studies", total=len(studies_list)):
    study_path = os.path.join(studies_dir, study)
    txt_files = [f for f in os.listdir(study_path) if f.endswith('.txt')]
    if len(txt_files) == 0:
        studies_with_no_report.append(study)
    else:
        for file in txt_files:
            file_path = os.path.join(study_path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(min(len(lines), 5)):
                    if "Codes" in lines[i]:
                        report_types[study] = [line.strip() for line in lines[i+1].split("\n")[0].split('.') if line.strip()]
                        type_set.update(report_types[study])
                        break
                    # if i == 4:
                    #     cnt_abnormal += 1
                    #     abnormal_report.append("".join(lines[:4]))
special_type_set = dict()

for types in ["Stress", "Transesophageal", "TEE", "congenital", "Stress/Transesophageal/TEE", "Stress/Transesophageal/TEE/Congenital", "Transthoracic/TTE"]:
    special_type_set[types] = set()
    for t in types.split('/'):
        for study, types_study in report_types.items():
            for type_study in types_study:
                if t.lower() in type_study.lower():
                    special_type_set[types].add(study)
                    break
            # special_type_set[types].update([study for study, type_study in report_types.items() if t in type_study])
type_num = {types: len(special_type_set[types]) for types in special_type_set.keys()}
# type_num = {types: len(set{study for study, type_study in report_types.items() for t in types.split('/') if t in type_study}) for types in ["Stress", "Transesophageal", "TEE", "congenital", "Stress/Transesophageal/TEE/congenital"]}

type_num = type_num | {t: len([study for study, types in report_types.items() if t in types]) for t in sorted(list(type_set))}

# abnormal_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/report_abnormal.json"
type_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/report_type.json"
type_num_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/report_type_num.json"
studies_with_no_report_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_no_report.json"
# with open(title_path, 'w') as f:
#     json.dump(title_list, f, indent=4)
#     print(f"🎉🎉🎉 Title list saved successfully to {title_path}. 🎉🎉🎉")
with open(type_path, 'w') as f:
    # json.dump(sorted(list(type_set)), f, indent=4)
    json.dump(report_types, f, indent=4)
    print(f"🎉🎉🎉 Type list saved successfully to {type_path}. 🎉🎉🎉")
with open(type_num_path, 'w') as f:
    json.dump(type_num, f, indent=4)
    print(f"🎉🎉🎉 Type num saved successfully to {type_num_path}. 🎉🎉🎉")
with open(studies_with_no_report_path, 'w') as f:
    json.dump(studies_with_no_report, f, indent=4)
    print(f"🎉🎉🎉 Studies with no report saved successfully to {studies_with_no_report_path}. 🎉🎉🎉")
# with open(abnormal_path, 'w') as f:
#     json.dump(abnormal_report, f, indent=4)
#     print(f"🎉🎉🎉 Abnormal report saved successfully to {abnormal_path}. 🎉🎉🎉")
    