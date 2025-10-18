import pandas as pd
import os

# dataset_name = "zeroshot"
dataset_name = "surgery_indication_v0"

# result_dir = f"/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_{dataset_name}/"
             # "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_mini_study_only/",
             # "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_mini_new_study_only/",
# result_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Zero_shot_result/"
result_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_study_only/"
# result_dirs = ["/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_results_clip_mini/"]

output_dir = f"/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/{dataset_name}_bootstrap_crossval/Results"

severity = False
no_freeze = False
# key_word = "epoch100_new"
key_word = "crossval"
key_word1 = "surgery_indication"
# no_word = "backup"
no_word = None

def get_model_name(run):
    if "echoclip" in run:
        return "EchoCLIP"
    elif "biomedclip" in run:
        return "BioMedCLIP"
    elif "biomedgpt" in run:
        return "BiomedGPT"
    elif "transformer_original1" in run:
        return "DINOv2_transformer_original1"
    elif "transformer" in run:
        return "DINOv2_transformer"
    elif "study_original1" in run:
        return "DINOv2_study_original1"
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

def data_name(run):
    if "fullsize" in run:
        return "Original Size"
    else:
        return "Processed Size"

results = {}

print(f"Processing results in {result_dir}")
runs = [result_dir + run for run in os.listdir(result_dir)]
for i in range(5):
    runs = [run for run in runs if os.path.isdir(run + f"/fold_{i}") and "results_bootstrap.csv" in os.listdir(run + f"/fold_{i}")]
# runs = [run for run in runs if os.path.isdir(run) and "results_bootstrap.csv" in os.listdir(run)]
# runs = [run for run in runs if "epoch" in run]
if no_freeze:
    runs = [run for run in runs if "freeze" not in run]
if severity:
    runs = [run for run in runs if "severity" in run]
else:
    runs = [run for run in runs if "severity" not in run]
if key_word:
    runs = [run for run in runs if key_word in run or "echoclip_epoch100" in run]
if key_word1:
    runs = [run for run in runs if key_word1 in run]
if no_word:
    runs = [run for run in runs if no_word not in run]

print(f"Found {len(runs)} runs in {result_dir}")
# result_files = [run + "/results_bootstrap.csv" for run in runs]
# print(f"Found {len(result_files)} result files in {result_dir}")

# for result_file in result_files:
#     print(f"Reading {result_file}")
#     df = pd.read_csv(result_file)
    
#     # Check if "AUROC" column exists
#     if "AUROC" in df.columns:
#         run = result_file.split("/")[-2]  # Add the run name to the row
#         task = run.split("_clip")[0] if "_clip" in run else run.split("_mini")[0]
#         if task not in results:
#             results[task] = {}
        
#         model = get_model_name(run)
#         result = {}
#         for col in df.columns:
#             result[col] = df[col].to_numpy()
#         results[task][model] = result
#     else:
#         print(f"Skipping {result_file} as it does not contain 'AUROC' column.")
#         continue
    
#     output_file = os.path.join(output_dir, task, f"{model}.csv")
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     df.to_csv(output_file, index=False)

for run in runs:
    print(f"Processing run: {run}")
    
    # Get run name and derive task and model
    run_name = os.path.basename(run)
    task = run_name.split("_clip")[0] if "_clip" in run_name else run_name.split("_mini")[0]
    model = get_model_name(run_name)
    
    # Collect all fold results
    fold_dfs = []
    for i in range(5):
        fold_file = os.path.join(run, f"fold_{i}", "results_bootstrap.csv")
        if os.path.exists(fold_file):
            print(f"  Reading {fold_file}")
            df = pd.read_csv(fold_file)
            
            # Add fold identifier
            df['fold'] = i
            fold_dfs.append(df)
        else:
            print(f"  Warning: Missing {fold_file}")
    
    if len(fold_dfs) == 0:
        print(f"  Skipping {run} - no valid fold files found")
        continue
    
    # Combine all folds
    combined_df = pd.concat(fold_dfs, ignore_index=True)
    
    # Check if "AUROC" column exists
    if "AUROC" not in combined_df.columns:
        print(f"  Skipping {run} - no AUROC column found")
        continue
    
    # Save combined results
    output_file = os.path.join(output_dir, task, f"{model}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"  Saving combined results to {output_file}")
    combined_df.to_csv(output_file, index=False)
    
    # Store in results dictionary for further processing if needed
    if task not in results:
        results[task] = {}
    
    result = {}
    for col in combined_df.columns:
        result[col] = combined_df[col].to_numpy()
    results[task][model] = result

print("Processing completed!")
print(f"Results saved to {output_dir}")