import os
import pandas as pd

# method_dict = {
#     "DINOv2+CLIP": ["DINOv2_transformer", "DINOv2_transformer_original1", "DINOv2_study","DINOv2_study_original1" , "DINOv2_clip"],
#     "DINOv2+pretrained": ["DINOv2_original", "DINOv2_original1"],
#     "DINOv2_public": ["DINOv2_public"],
#     "EchoCLIP": ["EchoCLIP"]
# }

# method_dict = {
#     "DINOv2+CLIP": ["DINOv2_study"],
#     "DINOv2+pretrained": ["DINOv2_original"],
#     "DINOv2_public": ["DINOv2_public"],
#     "EchoCLIP": ["EchoCLIP"]
# }

method_dict = {
    "DINOv2+CLIP": ["DINOv2_study_original1"],
    "DINOv2+pretrained": ["DINOv2_original1"],
    "DINOv2_public": ["DINOv2_public"],
    "EchoCLIP": ["EchoCLIP"]
}

# method_dict = {
#     "DINOv2+CLIP": ["DINOv2_study_original1"],
#     "EchoCLIP": ["EchoCLIP"],
#     "BioMedCLIP": ["BioMedCLIP"],
#     "BiomedGPT": ["BiomedGPT"]
# }

# method_dict = {
#     "DINOv2+CLIP": ["DINOv2_study"],
#     "EchoCLIP": ["EchoCLIP"],
#     "BioMedCLIP": ["BioMedCLIP"],
#     "BiomedGPT": ["BiomedGPT"]
# }

input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/surgery_indication_v0_bootstrap/Results"
output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/surgery_indication_v0_bootstrap_processed/Results"
# input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/surgery_indication_v0_bootstrap_crossval/Results"
# output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/surgery_indication_v0_bootstrap_crossval_processed/Results"
# input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/study_only_v1_bootstrap_crossval/Results"
# output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/study_only_v1_bootstrap_crossval_processed/Results"
# input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/study_only_v1_bootstrap/Results"
# output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/study_only_v1_bootstrap_processed/Results"
# input_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/zeroshot_v1_bootstrap/Results"
# output_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/Plot/zeroshot_v1_bootstrap_processed/Results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for task in os.listdir(input_dir):
    task_path = os.path.join(input_dir, task)
    
    for method in method_dict:
        methods = method_dict[method]
        max_auroc = 0
        best_df = None
        
        for m in methods:
            results_file = os.path.join(task_path, m + ".csv")
            
            df = pd.read_csv(results_file)
            assert "AUROC" in df.columns, f"AUROC column not found in {results_file}"
            if df["AUROC"].mean() > max_auroc:
                max_auroc = df["AUROC"].mean()
                best_df = df
                
        assert best_df is not None, f"No valid results found for {task} with methods {methods}"
        os.makedirs(os.path.join(output_dir, task), exist_ok=True)
        output_file = os.path.join(output_dir, task, method + ".csv")
        best_df.to_csv(output_file, index=False)
        print(f"Processed {task} for method {method}, saved to {output_file}")
    