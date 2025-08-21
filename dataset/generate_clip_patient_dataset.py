import pandas as pd
import os
import json

input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/clip_study/"
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_dataset_csv/clip_patient/"
video_dict_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos_dict.json"

def get_patient_id(study_video_path):
    return study_video_path.split('/')[-2]

for file_name in os.listdir(input_dir):
    if not file_name.endswith('.csv'):
        continue
    print(f"Processing file: {file_name}")
    
    df = pd.read_csv(os.path.join(input_dir, file_name))
    
    new_df = pd.DataFrame(columns=['study', 'report']) if 'report' in df.columns else pd.DataFrame(columns=['study', 'report_path'])
    new_df['study'] = df['video_path'].apply(lambda x: x.split('/')[-2])
    if 'report' in df.columns:
        new_df['report'] = df['report']
    else:
        new_df['report_path'] = df['report_path']
        
    # Save the new DataFrame to a CSV file
    output_file_path = os.path.join(output_dir, file_name)
    new_df.to_csv(output_file_path, index=False)
    
    out_dict = {"report_data": output_file_path, "video_data": video_dict_path}
    output_dict_path = os.path.join(output_dir, file_name.replace('.csv', '.json'))
    with open(output_dict_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
    
    print(f"Saved processed data to {output_file_path} and {output_dict_path}")