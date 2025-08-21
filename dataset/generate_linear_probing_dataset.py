import pandas as pd

dataset_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_mini/"
task_set = ['AV_regurgitation_severity', 'AV_regurgitation', 'AV_stenosis_severity', 'AV_stenosis', 'AV_vegetations', 'DF_severity', 'DF', 'IMT', 'IS', 'LAD_severity', 'LAD', 'LHF_severity', 'LHF', 'LVD_severity', 'LVD', 'LVH_severity', 'LVH', 'MV_regurgitation_severity', 'MV_regurgitation', 'MV_stenosis_severity', 'MV_stenosis', 'MV_vegetations', 'PE_severity', 'PE', 'PV_regurgitation_severity', 'PV_regurgitation', 'PV_stenosis_severity', 'PV_stenosis', 'PV_vegetations', 'RAD_severity', 'RAD', 'RHF_severity', 'RHF', 'RVD_severity', 'RVD', 'TAPSE_severity', 'TV_regurgitation_severity', 'TV_regurgitation', 'TV_stenosis_severity', 'TV_stenosis', 'TV_vegetations']
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_mini_study_only/"

def get_study(str):
    return str.split('/')[0]

for task in task_set:
    print(f"Processing task: {task}")
    df = pd.read_csv(f"{dataset_dir}{task}.csv")
    
    # Filter out rows with NaN values in the 'label' column
    df["path"] = df["path"].apply(get_study)
    
    # Save the filtered DataFrame to a new CSV file
    output_file = f"{output_dir}{task}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Filtered dataset saved to {output_file}")