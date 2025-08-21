from open_clip import create_model_and_transforms, tokenize
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
import time

# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP for zero-shot tasks like ejection fraction prediction
# or pacemaker detection. It has a short context window because it
# uses the CLIP BPE tokenizer, so it can't process an entire report at once.

labels = {
    'LHF': ['0', '1', '0.0', '1.0'],
    'RHF': ['0', '1', '0.0', '1.0'],
    'DF': ['0', '1', '0.0', '1.0'],
    'LAD': ['0', '1', '0.0', '1.0'],
    'LVD': ['0', '1', '0.0', '1.0'],
    'RAD': ['0', '1', '0.0', '1.0'],
    'RVD': ['0', '1', '0.0', '1.0'],
    # 'AVA': ['0', '1', '0.0', '1.0'],
    # 'MVA': ['0', '1', '0.0', '1.0'],
    # 'TVA': ['0', '1', '0.0', '1.0'],
    # 'PVA': ['0', '1', '0.0', '1.0'],
    'PE': ['0', '1', '0.0', '1.0'],
    'LVH': ['0', '1', '0.0', '1.0'],
    'IMT': ['0', '1', '0.0', '1.0'],
    'IS': ['0', '1', '0.0', '1.0'],
    'AV_regurgitation': ['0', '1', '0.0', '1.0'],
    'AV_stenosis': ['0', '1', '0.0', '1.0'],
    'AV_vegetations': ['0', '1', '0.0', '1.0'],
    'MV_regurgitation': ['0', '1', '0.0', '1.0'],
    'MV_stenosis': ['0', '1', '0.0', '1.0'],
    'MV_vegetations': ['0', '1', '0.0', '1.0'],
    'PV_regurgitation': ['0', '1', '0.0', '1.0'],
    'PV_stenosis': ['0', '1', '0.0', '1.0'],
    'PV_vegetations': ['0', '1', '0.0', '1.0'],
    'TV_regurgitation': ['0', '1', '0.0', '1.0'],
    'TV_stenosis': ['0', '1', '0.0', '1.0'],
    'TV_vegetations': ['0', '1', '0.0', '1.0'],
    'LHF_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RHF_severity': ['normal', 'mild', 'moderate', 'severe'],
    'DF_severity': ['normal', 'grade 1', 'grade 2', 'grade 3'],
    'LAD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'LVD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RAD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RVD_severity': ['normal', 'mild', 'moderate', 'severe'],
    # 'AVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    # 'MVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    # 'TVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    # 'PVA_severity': ['normal', 'mild', 'moderate', 'severe'],
    'PE_severity': ['small', 'moderate', 'large', 'tamponade physiology'],
    'LVH_severity': ['normal', 'mild', 'moderate', 'severe'],
    'AV_regurgitation_severity': ['normal', 'mild', 'moderate', 'severe'],
    'AV_stenosis_severity': ['normal', 'mild', 'moderate', 'severe'],
    'MV_regurgitation_severity': ['normal', 'mild', 'moderate', 'severe'],
    'MV_stenosis_severity': ['normal', 'mild', 'moderate', 'severe'],
    'PV_regurgitation_severity': ['normal', 'mild', 'moderate', 'severe'],
    'PV_stenosis_severity': ['normal', 'mild', 'moderate', 'severe'],
    'TV_regurgitation_severity': ['normal', 'mild', 'moderate', 'severe'],
    'TV_stenosis_severity': ['normal', 'mild', 'moderate', 'severe'],
    'TAPSE_severity': ['normal', 'mild', 'moderate', 'severe'],
}

# tasks without prompt: limited, DF_severity
task_prompts = {
    "AV_regurgitation_severity": ["* aortic valve regurgitation. "],
    "AV_regurgitation": ["aortic valve regurgitation. "],
    "AV_stenosis_severity": ["* aortic valve stenosis. "],
    "AV_stenosis": ["aortic valve stenosis. "],
    "AV_vegetations": ["aortic valve valve vegetation. "],
    "DF": ["diastolic dysfunction. "],
    "IMT": ["intracardiac masses and thrombi / singular thrombus. "],
    "IS": ["intracavitary shunts. "],
    "LAD_severity": ["* left atrial dilation. "],
    "LAD": ["left atrial dilation. "],
    "LHF_severity": ["* left heart failure. "],
    "LHF": ["left heart failure. "],
    "LVD_severity": ["* left ventricular dilation. "],
    "LVD": ["left ventricular dilation. "],
    "LVH_severity": ["* left ventricular hypertrophy. "],
    "LVH": ["left ventricular hypertrophy. "],
    "MV_regurgitation_severity": ["* mitral valve regurgitation. "],
    "MV_regurgitation": ["mitral valve regurgitation. "],
    "MV_stenosis_severity": ["* mitral valve stenosis. "],
    "MV_stenosis": ["mitral valve stenosis. "],
    "MV_vegetations": ["mitral valve vegetation. "],
    "PE_severity": ["pericardial effusion size is *. "],
    "PE": ["pericardial effusion. "],
    "PV_regurgitation_severity": ["* pulmonic valve regurgitation. "],
    "PV_regurgitation": ["pulmonic valve regurgitation. "],
    "PV_stenosis_severity": ["* pulmonic valve stenosis. "],
    "PV_stenosis": ["pulmonic valve stenosis. "],
    "PV_vegetations": ["pulmonic valve vegetation. "],
    "RAD_severity": ["* right atrium dilation. ", "* right atrial dilation. "],
    "RAD": ["right atrium dilation. ", "right atrial dilation. "],
    "RHF_severity": ["* right heart failure. "],
    "RHF": ["right heart failure. "],
    "RVD_severity": ["* right ventricular dilation. ", "* right ventricle dilation. "],
    "RVD": ["right ventricular dilation. ", "right ventricle dilation. "],
    "TAPSE_severity": ["* tricuspid annular plane systolic excursion. "],
    "TV_regurgitation_severity": ["* tricuspid valve regurgitation. "],
    "TV_regurgitation": ["tricuspid valve regurgitation. "],
    "TV_stenosis_severity": ["* tricuspid valve stenosis. "],
    "TV_stenosis": ["tricuspid valve stenosis. "],
    "TV_vegetations": ["tricuspid valve vegetations. "],
}

prompts_DF_severity = {
    "normal": ["diastolic dysfunction. "],
    "grade 1": ["Impaired relaxation pattern with reduced mitral annular e' velocity and prolonged deceleration time. "],
    "grade 2": ["Pseudonormal mitral inflow with elevated E/e' ratio and left atrial enlargement. "],
    "grade 3": ["Restrictive filling with short deceleration time and severely elevated E/e'. "],
}

def save_studies(task_set, csv_dir, save_path, **kwargs):
    """
    Save the studies that need to compute embeddings in a json file.
    """
    studies = set()
    for task in task_set:
        print(f"Processing {task}...")
        df = pd.read_csv(csv_dir + f"{task}.csv", dtype=str)
        for (path, label) in zip(list(df["path"]), list(df["label"])):
            if label in labels[task] and label not in ['normal', '0', '0.0']:
                studies.add(path)
    
    studies = sorted(list(studies))
    with open(save_path, "w") as f:
        json.dump(studies, f, indent=4)
        print(f"Saved {len(studies)} studies to {save_path}")

@torch.no_grad()
def compute_embeddings(echo_clip, studies_path, embedding_dir, dataset_dir, start, end, step, device, **kwargs):
    """
    Compute the embeddings for every video in the studies in the pt file.
    """

    mean_tensor = torch.tensor(echo_clip.visual.image_mean or (0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(echo_clip.visual.image_std or (0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1).to(device)
    
    with open(studies_path, "r") as f:
        studies = json.load(f)
    studies = studies[start: end: step]
    
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    
    for study in tqdm(studies, total=len(studies)):
        embeddings = dict()
        if os.path.exists(embedding_dir + f"{study}.pt"):
            print(f"Embedding for {study} already exists, skipping...")
            continue
        video_dir = dataset_dir + study
        with tqdm(total=len(os.listdir(video_dir)), desc="Processing Study") as pbar:
            for file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, file)
                short_path = os.path.join(study, file)
                
                test_video = torch.from_numpy(np.load(video_path)).movedim(0, 1)[::2].to(device)
                test_video = F.interpolate(test_video.to(torch.float32), size=echo_clip.visual.image_size, mode="bicubic", align_corners=False) / 255.0
                test_video = (test_video - mean_tensor) / std_tensor
                test_video = test_video.to(torch.bfloat16)
                
                test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1)
                embeddings[short_path] = test_video_embedding.mean(dim=0)
                
                pbar.update(1)
                
        torch.save(embeddings, embedding_dir + f"{study}.pt")
        print(f"Embeddings saved to {embedding_dir + f'{study}.pt'}")
    print("🎉🎉🎉 All embeddings saved! 🎉🎉🎉")

def check_progress(studies_path, embedding_dir, **kwargs):
    with open(studies_path, "r") as f:
        studies = json.load(f)
    processed = False
    start = 0
    processed_interval = []
    cnt = 0
    
    for idx, study in enumerate(studies):
        if os.path.exists(embedding_dir + f"{study}.pt") and not processed:
            processed = True
            start = idx
        if not os.path.exists(embedding_dir + f"{study}.pt") and processed:
            processed = False
            processed_interval.append([start, idx - 1])
            cnt += (idx - start)
    
    if processed:
        processed_interval.append([start, len(studies) - 1])
        cnt += (len(studies) - start)
    
    print(f"Processed intervals: {processed_interval}")
    print(f"Total processed studies: {cnt}/ {len(studies)}")

def merge(merged_embedding_path, embedding_dir, **kwargs):
    """
    merge the embeddings into one file with structure:
    {
        "study1": {
            "study1/video1": embedding1,
            "study1/video2": embedding2,
            ...
        },
        "study2": {
            "study2/video1": embedding1,
            "study2/video2": embedding2,
            ...
        },
        ...
    }

    Args:
        merged_embedding_path (str): path to the output merged file
        embedding_dir (str): directory containing the embedding files
    """
    if os.path.exists(merged_embedding_path):
       merged_embeddings = torch.load(merged_embedding_path)
    else:
       merged_embeddings = dict()
    print(f"✊✊✊ Start merging embeddings from {embedding_dir} ✊✊✊")
    for file in tqdm(os.listdir(embedding_dir), total=len(os.listdir(embedding_dir))):
        assert file.endswith(".pt")
        study = file.replace(".pt", "")
        
        embeddings = torch.load(os.path.join(embedding_dir, file))
        merged_embeddings[study] = embeddings
        
        # if (idx + 1) % 100 == 0:
        #     print(f"Processed {idx + 1} files")
    print(f"🎉🎉🎉 Finish merging embeddings 🎉🎉🎉")
    start_time = time.time()
    torch.save(merged_embeddings, merged_embedding_path)
    end_time = time.time()
    print(f"Time taken to save merged embeddings: {end_time - start_time} seconds")
    print(f"🎉🎉🎉 Merged embeddings saved to {merged_embedding_path} 🎉🎉🎉")

@torch.no_grad()
def dataset_filtering(echo_clip, task_set, merged_embedding_path, csv_dir, new_csv_dir, dataset_dir, **kwargs):
    """
    Filter the dataset to only include the videos that have largest similarity with prompts in each studies for each positive sample, and a randomly chosen video for each negative sample, output into a csv file.

    Args:
        merged_embedding_path (str): path to the merged embedding file
        csv_dir (str): path to the original csv directory
        new_csv_dir (str): path to the new dataset directory
    """
    print(f"✊✊✊ Loading merged embeddings from {merged_embedding_path} ✊✊✊")
    merged_embeddings = torch.load(merged_embedding_path)
    print(f"🎉🎉🎉 Finish loading merged embeddings 🎉🎉🎉")
    
    if not os.path.exists(new_csv_dir):
        os.makedirs(new_csv_dir)
    
    for idx, task in enumerate(task_set):
        
        if os.path.exists(new_csv_dir + f"{task}.csv"):
            print(f"Filtered dataset for {task} already exists, skipping...")
            continue
        
        print(f"{idx + 1}/{len(task_set)} 🤔🤔🤔 Processing {task}... 😊😊😊")
        df = pd.read_csv(csv_dir + f"{task}.csv", dtype=str)
        out_data = {"label": list(), "split": list(), "path": list()}
        
        label_to_prompts = dict()
        label_to_prompt_embedding = dict()
        
        for label in labels[task]:
            assert label in labels[task]
            
            if task != 'DF_severity':
                if label in ['normal', '0', '0.0']:
                    # give all negative samples a positive prompt
                    label_to_prompts[label] = [prompt.replace('* ', '').upper() for prompt in task_prompts[task]]
                else:
                    label_to_prompts[label] = [prompt.replace('*', label).upper() for prompt in task_prompts[task]]
            else:
                label_to_prompts[label] = prompts_DF_severity[label].upper()
                
            label_to_prompt_embedding[label] = F.normalize(echo_clip.encode_text(tokenize(label_to_prompts[label]).to(device)), dim=-1)
        
        max_similarity_sum = 0
        max_similarity_count = 0
        all_similarity = 0
        all_similarity_count = 0
        with tqdm(total=len(df['label']), desc=f"Processing {task}") as pbar:
            for (path, label, split) in zip(list(df["path"]), list(df["label"]), list(df["split"])):
                
                assert label in labels[task]
                assert path in merged_embeddings
                
                max_similarity = 0
                max_video = None
                prompt_embedding = label_to_prompt_embedding[label] # [n, 512]
                for video_path, video_embedding in merged_embeddings[path].items():
                    video_embedding = video_embedding.to(device) # [1, 512]
                    similarity = (video_embedding @ prompt_embedding.T).mean(dim=-1).item()
                    
                    assert similarity >= 0, f"similarity = {similarity} is incorrect"
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_video = video_path
                    
                    all_similarity += similarity
                    all_similarity_count += 1
                    
                assert max_video is not None
                out_data["label"].append(label)
                out_data["split"].append(split)
                out_data["path"].append(max_video)
                
                max_similarity_sum += max_similarity
                max_similarity_count += 1
                    
                pbar.update(1)
                pbar.set_postfix({
                    "avg max_similarity": f"{max_similarity_sum / max_similarity_count:.4f}" if max_similarity_count > 0 else "N/A", "similarity_count": max_similarity_count,
                    "avg all_similarity": f"{all_similarity / all_similarity_count:.4f}" if all_similarity_count > 0 else "N/A",
                    "all_similarity_count": all_similarity_count,
                })
        out_data = pd.DataFrame(data=out_data)
        out_data.to_csv(new_csv_dir + f"{task}.csv", index=None)
        print(f"🎉🎉🎉 Finish processing {task} ✌✌✌")
    

def argparse_args():
    """
    Parse the arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Start and end index of the studies to process")
    parser.add_argument("--start", type=int, default=0, help="Start index of the studies to process")
    parser.add_argument("--end", type=int, default=0, help="End index of the studies to process")
    parser.add_argument("--step", type=int, default=1, help="Step size of the studies to process")
    parser.add_argument("--csv_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/", help="Path to the csv directory")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/", help="Path to the dataset directory")
    parser.add_argument("--studies_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies.json", help="Path to the studies json file")
    parser.add_argument("--embedding_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/embedding_v4/", help="Path to the embedding file")
    parser.add_argument("--merged_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/merged_embedding_all_v4.pt", help="Path to the output merged file")
    parser.add_argument("--new_csv_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4/", help="Path to the new dataset csv directory")
    
    parser.add_argument("--save_studies", action="store_true", default=False, help="Save the studies that need to compute embeddings")
    parser.add_argument("--compute_embeddings", action="store_true", default=False, help="Compute the embeddings for every video in the studies in the pt file")
    parser.add_argument("--check_progress", action="store_true", default=False, help="check_progress the progress of the studies")
    parser.add_argument("--merge", action="store_true", default=False, help="Merge the embeddings into one file")
    parser.add_argument("--dataset_filtering", action="store_true", default=False, help="Filter the dataset to only include the videos that have largest similarity with prompts in each studies, output into a csv file")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for computation")
    return parser.parse_args()

# task_set = ['AV_regurgitation_severity', 'AV_regurgitation', 'AV_stenosis_severity', 'AV_stenosis', 'AV_vegetations', 'DF_severity', 'DF', 'IMT', 'IS', 'LAD_severity', 'LAD', 'LHF_severity', 'LHF', 'LVD_severity', 'LVD', 'LVH_severity', 'LVH', 'MV_regurgitation_severity', 'MV_regurgitation', 'MV_stenosis_severity', 'MV_stenosis', 'MV_vegetations', 'PE_severity', 'PE', 'PV_regurgitation_severity', 'PV_regurgitation', 'PV_stenosis_severity', 'PV_stenosis', 'PV_vegetations', 'RAD_severity', 'RAD', 'RHF_severity', 'RHF', 'RVD_severity', 'RVD', 'TAPSE_severity', 'TV_regurgitation_severity', 'TV_regurgitation', 'TV_stenosis_severity', 'TV_stenosis', 'TV_vegetations']
task_set = ['DF_severity']

csv_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/"
dataset_dir = "/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/"

args = argparse_args()

echo_clip = None
if args.compute_embeddings or args.dataset_filtering:
    device = torch.device(args.device)
    print("✊✊✊ Start loading model ✊✊✊")
    echo_clip, _, _ = create_model_and_transforms("hf-hub:mkaichristensen/echo-clip", precision="bf16", device=device)
    print("🎉🎉🎉 Finish loading model 🎉🎉🎉")

if args.save_studies:
    save_studies(task_set, **vars(args))
if args.compute_embeddings:
    compute_embeddings(echo_clip, **vars(args))
if args.check_progress:
    check_progress(**vars(args))
if args.merge:
    merge(**vars(args))
if args.dataset_filtering:
    dataset_filtering(echo_clip, task_set, **vars(args))