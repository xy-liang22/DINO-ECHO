import pandas as pd

label_dict_1 = {"normal": "0-normal", "mild": "1-mild", "moderate": "2-moderate", "severe": "3-severe"}
label_dict_2 = {"normal": "grade 0", "grade 1": "grade 1", "grade 2": "grade 2", "grade 3": "grade 3"}
label_dict_3 = {"small": "0-small", "moderate": "1-moderate", "large": "2-large", "tamponade physiology": "3-tamponade physiology"}

def process_labels(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Create a new DataFrame with unique labels and their corresponding indices
    unique_labels = df['label'].unique()
    print(f"Unique labels before mapping: {unique_labels}")
    
    if "mild" in unique_labels:
        label_dict = label_dict_1
    elif "grade 1" in unique_labels:
        label_dict = label_dict_2
    elif "small" in unique_labels:
        label_dict = label_dict_3

    # Map the labels to their corresponding indices
    df['label'] = df['label'].map(label_dict)
    print(f"Unique labels after mapping: {df['label'].unique()}")

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_mini_new_study_only_no_mapping/"
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4_clip_mini_new_study_only/"
task_set = ['AV_regurgitation_severity', 'AV_stenosis_severity','DF_severity', 'LAD_severity', 'LHF_severity', 'LVD_severity', 'LVH_severity', 'MV_regurgitation_severity', 'MV_stenosis_severity', 'PE_severity', 'PV_regurgitation_severity', 'PV_stenosis_severity', 'RAD_severity', 'RHF_severity', 'RVD_severity', 'TAPSE_severity', 'TV_regurgitation_severity', 'TV_stenosis_severity']

for task in task_set:
    input_file = f"{input_dir}{task}.csv"
    output_file = f"{output_dir}{task}.csv"
    print(f"Processing {task}... 😊😊😊")
    process_labels(input_file, output_file)
    