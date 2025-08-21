import json

original_file = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos_dict_full_path.json"
new_file = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos_dict_full_path_original.json"

with open(original_file, 'r') as f:
    data = json.load(f)

for study in data:
    for i in range(len(data[study])):
        data[study][i] = data[study][i].replace("20250126", "original_size")
        
with open(new_file, 'w') as f:
    json.dump(data, f, indent=4)
print(f"Updated file saved to {new_file}")