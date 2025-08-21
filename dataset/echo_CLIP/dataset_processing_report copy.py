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
from utils import get_all_studies

# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP for zero-shot tasks like ejection fraction prediction
# or pacemaker detection. It has a short context window because it
# uses the CLIP BPE tokenizer, so it can't process an entire report at once.

original_size_dir = "/mnt/hanoverdev/data/hanwen/ECHO/original_size/"

def not_regular_report(study, file):
    """
    Check if the report is not a regular report.
    """
    with open(os.path.join(original_size_dir, study, file), "r") as f:
        lines = f.readlines()
        for i in range(min(len(lines), 5)):
            if "Codes" in lines[i]:
                if "Stress" in lines[i + 1]:
                    return True
    return False


def save_studies(task_set, csv_dir, studies_path, raw_dataset_dir, **kwargs):
    """
    Save all the studies in all tasks with corresponding report in a json file.
    """
    studies_with_video = set()
    for task in task_set:
        print(f"Processing {task}...")
        df = pd.read_csv(os.path.join(csv_dir, f"{task}.csv"), dtype=str)
        for path in list(df["path"]):
            studies_with_video.add(path)
    
    studies_with_video = sorted(list(studies_with_video))
    studies_with_report = list()
    # exception = {"0": [], ">1": []}
    
    # enumerate each study to check whether its study type is regular
    with tqdm(total=len(studies_with_video), desc="Checking study types") as pbar:
        for study in tqdm(studies_with_video, desc="Checking study types", total=len(studies_with_video)):
            report_path_prefix = os.path.join(raw_dataset_dir, study)
            assert os.path.exists(report_path_prefix), f"Report path {report_path_prefix} does not exist"
            
            txt = [file for file in os.listdir(report_path_prefix) if file.endswith(".txt")]
            if len(txt):
                txt = sorted(txt)[-1]
                if not_regular_report(study, txt) == False:
                    studies_with_report.append(study)
            pbar.update(1)
            pbar.set_postfix({
                "studies_with_report": len(studies_with_report),
            })
                
    print(f"🎉🎉🎉 Finish checking reports of studies 🎉🎉🎉")
    
    with open(studies_path, "w") as f:
        json.dump(studies_with_report, f, indent=4)
        print(f"Saved {len(studies_with_report)} studies to {studies_path}")
        
    # exception_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/report_exception.json"
    # with open(exception_path, "w") as f:
    #     json.dump(exception, f, indent=4)
    #     print(f"Saved {len(exception['0'])} studies with 0 report and {len(exception['>1'])} studies with more than 1 report to {exception_path}")
        
        # print(f"Exception: {exception}")

@torch.no_grad()
def compute_video_embeddings(echo_clip, studies_path, video_embedding_dir, dataset_dir, start, end, step, device, **kwargs):
    """
    Compute the embeddings for every video in the studies in the pt file.
    """

    mean_tensor = torch.tensor(echo_clip.visual.image_mean or (0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(echo_clip.visual.image_std or (0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1).to(device)
    
    with open(studies_path, "r") as f:
        studies = json.load(f)
    studies = studies[start: end: step]
    
    if not os.path.exists(video_embedding_dir):
        os.makedirs(video_embedding_dir)
    
    for study in tqdm(studies, total=len(studies)):
        embeddings = dict()
        if os.path.exists(os.path.join(video_embedding_dir, f"{study}.pt")):
            print(f"Embedding for {study} already exists, skipping...")
            continue
        video_dir = os.path.join(dataset_dir, study)
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
                
        torch.save(embeddings, os.path.join(video_embedding_dir, f"{study}.pt"))
        print(f"Embeddings saved to {os.path.join(video_embedding_dir, f'{study}.pt')}")
    print("🎉🎉🎉 All embeddings saved! 🎉🎉🎉")

def compute_report_embeddings(echo_clip, studies_path, labels_path, report_embedding_path, raw_dataset_dir, start, end, step, device, **kwargs):
    """
    Compute the embeddings for the report in the studies in the pt file.

    Args:
        echo_clip (nn.Module): EchoClip model
        studies_path (str): path to the studies json file
        report_embedding_path (str): path to save the report embeddings
        raw_dataset_dir (str): path to the report directory
    """
    with open(studies_path, "r") as f:
        studies = json.load(f)
    with open(labels_path, "r") as f:
        studies_with_labels = json.load(f)
        
    studies = studies[start: end: step]
    studies_with_labels = {study: labels for study, labels in studies_with_labels.items() if study in studies}
    
    if os.path.exists(report_embedding_path):
        print(f"Report embedding path {report_embedding_path} already exists. Loading...")
        report_embeddings = torch.load(report_embedding_path)
    else:
        report_embeddings = dict()
        
    print(f"✊✊✊ Start computing report embeddings ✊✊✊")
    
    for study in tqdm(studies, total=len(studies)):
        report_path_prefix = os.path.join(raw_dataset_dir, study)
        assert os.path.exists(report_path_prefix), f"Report path {report_path_prefix} does not exist"
        
        prompts = []
        for file in os.listdir(report_path_prefix):
            if not file.endswith(".txt"):
                continue
            with open(os.path.join(report_path_prefix, file), "r") as f:
                report = f.read()
                prompts.append(clean_text(report))
        prompts = tokenize(prompts).to(device)
        report_embedding = F.normalize(echo_clip.encode_text(prompts), dim=-1)
        report_embeddings[file] = report_embedding
        
    torch.save(report_embeddings, report_embedding_path)
    print(f"🎉🎉🎉 Report embeddings saved to {report_embedding_path} 🎉🎉🎉")
    

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

def merge_video_embeddings(merged_video_embedding_path, embedding_dir, **kwargs):
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
        merged_video_embedding_path (_type_): path to the output merged file
        embedding_dir (_type_): directory containing the embedding files
    """
    if os.path.exists(merged_video_embedding_path):
       merged_embeddings = torch.load(merged_video_embedding_path)
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
    torch.save(merged_embeddings, merged_video_embedding_path)
    end_time = time.time()
    print(f"Time taken to save merged embeddings: {end_time - start_time} seconds")
    print(f"🎉🎉🎉 Merged embeddings saved to {merged_video_embedding_path} 🎉🎉🎉")

@torch.no_grad()
def video_filtering(echo_clip, studies_path, dataset_dir, raw_dataset_dir, save_path, start, end, step, device, **kwargs):
    """
    Filter the dataset to only include the videos that have largest similarity with its reports, output into a json file.

    Args:
        echo_clip (nn.Module): EchoClip model
        studies_path (str): path to the studies json file
        dataset_dir (str): path to the dataset directory
        raw_dataset_dir (str): path to the report directory
        save_path (str): path to save the filtered dataset
    """
    mean_tensor = torch.tensor(echo_clip.visual.image_mean or (0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(echo_clip.visual.image_std or (0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1).to(device)
    
    with open(studies_path, "r") as f:
        studies = json.load(f)
    studies = studies[start: end: step]
    
    if os.path.exists(save_path):
        print(f"Filtered dataset path {save_path} already exists. Loading...")
        filtered_dataset = torch.load(save_path)
    else:
        filtered_dataset = dict()
        
    print(f"✊✊✊ Start filtering dataset ✊✊✊")
    with tqdm(total=len(studies), desc=f"Filtering dataset") as pbar:
        max_similarity_sum = 0
        max_similarity_count = 0
        all_similarity_sum = 0
        all_similarity_count = 0
        for study in studies:
            report_path_prefix = os.path.join(raw_dataset_dir, study)
            assert os.path.exists(report_path_prefix), f"Report path {report_path_prefix} does not exist"
            
            # load the report and compute the embedding
            prompts = []
            for file in os.listdir(report_path_prefix):
                if not file.endswith(".txt"):
                    continue
                with open(os.path.join(report_path_prefix, file), "r") as f:
                    report = f.read()
                    prompts.append(clean_text(report))
            assert len(prompts) > 0, f"No report found for {study}"
            prompts = tokenize(prompts).to(device)
            report_embedding = F.normalize(echo_clip.encode_text(prompts), dim=-1) # [n, 512]
            
            video_dir = os.path.join(dataset_dir, study)
            assert os.path.exists(video_dir), f"Video directory {video_dir} does not exist"
            
            # select the one with the largest similarity
            max_similarity = 0
            max_video_path = None
            for file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, file)
                short_path = os.path.join(study, file)
                
                # load the video and compute the embedding
                test_video = torch.from_numpy(np.load(video_path)).movedim(0, 1)[::2].to(device)
                test_video = F.interpolate(test_video.to(torch.float32), size=echo_clip.visual.image_size, mode="bicubic", align_corners=False) / 255.0
                test_video = (test_video - mean_tensor) / std_tensor
                test_video = test_video.to(torch.bfloat16)
                test_video_embedding = F.normalize(echo_clip.encode_image(test_video), dim=-1).mean(dim=0, keepdim=True) # [1, 512]
                
                # compute the similarity
                similarity = (test_video_embedding @ report_embedding.T).mean(dim=-1).item()
                assert similarity >= 0, f"similarity = {similarity} is incorrect"
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_video_path = short_path
                        
                all_similarity_sum += similarity
                all_similarity_count += 1
                        
            assert max_video_path is not None
            filtered_dataset[study] = max_video_path
                    
            max_similarity_sum += max_similarity
            max_similarity_count += 1
                        
            pbar.update(1)
            pbar.set_postfix({
                "avg max_similarity": f"{max_similarity_sum / max_similarity_count:.4f}" if max_similarity_count > 0 else "N/A",
                "max_similarity_count": max_similarity_count,
                "avg all_similarity": f"{all_similarity_sum / all_similarity_count:.4f}" if all_similarity_count > 0 else "N/A",
                "similarity_all_count": all_similarity_count,
            })
        
    torch.save(filtered_dataset, save_path)
    print(f"🎉🎉🎉 Filtered dataset saved to {save_path} 🎉🎉🎉")
    

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
    parser.add_argument("--raw_dataset_dir", type=str, default="/mnt/hanoverdev/data/hanwen/ECHO/deidentified/", help="Path to the report directory")
    parser.add_argument("--studies_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_report.json", help="Path to the studies json file")
    parser.add_argument("--labels_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_labels.json", help="Path to the studies_with_labels json file")
    parser.add_argument("--video_embedding_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/video_embedding_v4/", help="Path to the video embedding file")
    parser.add_argument("--report_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/report_embedding_v4.pt", help="Path to the report embedding file")
    parser.add_argument("--merged_video_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/merged_video_embedding_v4.pt", help="Path to the output merged file")
    parser.add_argument("--new_csv_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4/", help="Path to the new dataset csv directory")
    
    parser.add_argument("--save_studies", action="store_true", default=False, help="Save the studies that need to compute embeddings")
    parser.add_argument("--compute_video_embeddings", action="store_true", default=False, help="Compute the embeddings for every video in the studies in the pt file")
    parser.add_argument("--compute_report_embeddings", action="store_true", default=False, help="Compute the embeddings for the report in the studies in the pt file")
    parser.add_argument("--check_progress", action="store_true", default=False, help="check_progress the progress of the studies")
    parser.add_argument("--merge_video_embeddings", action="store_true", default=False, help="Merge the embeddings into one file")
    parser.add_argument("--dataset_filtering", action="store_true", default=False, help="Filter the dataset to only include the videos that have largest similarity with prompts in each studies, output into a csv file")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for computation")
    return parser.parse_args()

task_set = ['AV_regurgitation_severity', 'AV_regurgitation', 'AV_stenosis_severity', 'AV_stenosis', 'AV_vegetations', 'DF_severity', 'DF', 'IMT', 'IS', 'LAD_severity', 'LAD', 'LHF_severity', 'LHF', 'LVD_severity', 'LVD', 'LVH_severity', 'LVH', 'MV_regurgitation_severity', 'MV_regurgitation', 'MV_stenosis_severity', 'MV_stenosis', 'MV_vegetations', 'PE_severity', 'PE', 'PV_regurgitation_severity', 'PV_regurgitation', 'PV_stenosis_severity', 'PV_stenosis', 'PV_vegetations', 'RAD_severity', 'RAD', 'RHF_severity', 'RHF', 'RVD_severity', 'RVD', 'TAPSE_severity', 'TV_regurgitation_severity', 'TV_regurgitation', 'TV_stenosis_severity', 'TV_stenosis', 'TV_vegetations']
# task_set = ['LHF']

labels_filtering = ['AV_regurgitation', 'AV_stenosis', 'AV_vegetations', 'DF', 'IMT', 'IS', 'LAD', 'LHF', 'LVD', 'LVH', 'MV_regurgitation', 'MV_stenosis', 'MV_vegetations', 'PE',  'PV_regurgitation', 'PV_stenosis', 'PV_vegetations', 'RAD', 'RHF', 'RVD', 'TAPSE_severity',  'TV_regurgitation', 'TV_stenosis', 'TV_vegetations']

csv_dir = "/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/"
dataset_dir = "/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/"

args = argparse_args()

echo_clip = None
if args.compute_video_embeddings or args.compute_report_embeddings or args.dataset_filtering:
    device = torch.device(args.device)
    print("✊✊✊ Start loading model ✊✊✊")
    # You'll need to log in to the HuggingFace hub CLI to download the models
    # You can do this with the terminal command "huggingface-cli login"
    # You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

    # Use EchoCLIP-R for retrieval-based tasks where you want to find
    # the similarity between two echos, like in patient identification or
    # echo report retrieval. It has a longer context window because it
    # uses the template tokenizer, which we found increases its retrieval
    # performance but decreases its performance on other zero-shot tasks.
    echo_clip_r, _, _ = create_model_and_transforms("hf-hub:mkaichristensen/echo-clip-r", precision="bf16", device=device)
    print("🎉🎉🎉 Finish loading model 🎉🎉🎉")

if args.save_studies:
    save_studies(task_set, **vars(args))
if args.compute_video_embeddings:
    compute_video_embeddings(echo_clip_r, **vars(args))
if args.check_progress:
    check_progress(**vars(args))
if args.merge_video_embeddings:
    merge_video_embeddings(task_set, echo_clip_r, **vars(args))
if args.dataset_filtering:
    video_filtering(echo_clip_r, task_set, **vars(args))