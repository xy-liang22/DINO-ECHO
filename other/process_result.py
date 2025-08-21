import pandas as pd
import os

result_dirs = [
               "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_study_only/",
            #    "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_mini_study_only/",
            #    "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_mini_new_study_only/",
               ]
# result_dirs = ["/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_mini/"]

severity = False
no_freeze = False
key_word = "epoch100_new"

combined_results = []

def get_model_name(run):
    if "echoclip" in run:
        return "EchoClip"
    elif "transformer_original1" in run:
        return "DINOv2_transformer_original1"
    elif "study_original1" in run:
        return "DINOv2_study_original1"
    elif "transformer" in run:
        return "DINOv2_transformer"
    elif "original" in run:
        if "original1" in run:
            return "DINOv2_original1"
        return "DINOv2_original"
    elif "public" in run:
        return "DINOv2_public"
    elif "study" in run:
        return "DINOv2_study"
    else:
        return "DINOv2_clip"

def get_dataset_name(result_dir):
    return result_dir.split("/")[-2].replace("ECHO_results_", "")

def data_name(run):
    if "fullsize" in run:
        return "Original Size"
    else:
        return "Processed Size"

for result_dir in result_dirs:
    print(f"Processing results in {result_dir}")
    runs = [result_dir + run for run in os.listdir(result_dir)]
    runs = [run for run in runs if os.path.isdir(run) and "results.csv" in os.listdir(run)]
    runs = [run for run in runs if "epoch" in run]
    if no_freeze:
        runs = [run for run in runs if "freeze" not in run]
    if severity:
        runs = [run for run in runs if "severity" in run]
    else:
        runs = [run for run in runs if "severity" not in run]
    if key_word:
        runs = [run for run in runs if key_word in run or "echoclip_epoch100" in run]
    print(f"Found {len(runs)} runs in {result_dir}")
    result_files = [run + "/results.csv" for run in runs]
    print(f"Found {len(result_files)} result files in {result_dir}")
    
    dataset = get_dataset_name(result_dir)
    
    for result_file in result_files:
        print(f"Reading {result_file}")
        df = pd.read_csv(result_file)
        
        # Check if "AUROC" column exists
        if "AUROC" in df.columns:
            # Select the row with the largest "AUROC"
            max_auroc_row = df.loc[df["AUROC"].idxmax()]
            run = result_file.split("/")[-2]  # Add the run name to the row
            max_auroc_row["model"] = get_model_name(run)
            max_auroc_row["task"] = run.split("_clip")[0] if "_clip" in run else run.split("_mini")[0]
            max_auroc_row["run"] = run
            max_auroc_row["dataset"] = dataset
            max_auroc_row["data_size"] = data_name(run)
            max_auroc_row["num_class"] = 2 if not severity else 4
            max_auroc_row["method"] = "linear probing" if "freeze" in run or "linear" in run else "fine-tuning"
            max_auroc_row["videos"] = "multi_videos" if "multi_videos" in run else "normal"
            combined_results.append(max_auroc_row)
        else:
            print(f"Skipping {result_file} as it does not contain 'AUROC' column.")

# Combine all rows into a single DataFrame
combined_df = pd.DataFrame(combined_results)

desired_order = ["model", "dataset", "task", "run", "data_size", "AUROC"]
columns = [col for col in combined_df.columns if col not in desired_order]
combined_df = combined_df[desired_order + columns]

# Save the combined results to a .csv file
output_file = "/mnt/hanoverdev/scratch/hanwen/xyliang/results_linear_probing_epoch100.csv"
print(f"Saving combined results to {output_file}")
combined_df.to_csv(output_file, index=False)