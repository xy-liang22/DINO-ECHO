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

labels = {
    'LHF': ['0', '1'],
    'RHF': ['0', '1'],
    'DF': ['0', '1'],
    'LAD': ['0', '1'],
    'LVD': ['0', '1'],
    'RAD': ['0', '1'],
    'RVD': ['0', '1'],
    'PE': ['0', '1'],
    'LVH': ['0', '1'],
    'IMT': ['0', '1'],
    'IS': ['0', '1'],
    'AV_regurgitation': ['0', '1'],
    'AV_stenosis': ['0', '1'],
    'AV_vegetations': ['0', '1'],
    'MV_regurgitation': ['0', '1'],
    'MV_stenosis': ['0', '1'],
    'MV_vegetations': ['0', '1'],
    'PV_regurgitation': ['0', '1'],
    'PV_stenosis': ['0', '1'],
    'PV_vegetations': ['0', '1'],
    'TV_regurgitation': ['0', '1'],
    'TV_stenosis': ['0', '1'],
    'TV_vegetations': ['0', '1'],
    'LHF_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RHF_severity': ['normal', 'mild', 'moderate', 'severe'],
    'DF_severity': ['normal', 'grade 1', 'grade 2', 'grade 3'],
    'LAD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'LVD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RAD_severity': ['normal', 'mild', 'moderate', 'severe'],
    'RVD_severity': ['normal', 'mild', 'moderate', 'severe'],
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

prompts = {
    'LHF':{'0': 'No left heart failure. ', '1': 'Left heart failure. '},
    'RHF':{'0': 'No right heart failure. ', '1': 'Right heart failure. '},
    'DF':{'0': 'No diastolic dysfunction. ', '1': 'Diastolic dysfunction. '},
    'LAD':{'0': 'No left atrial dilation. ', '1': 'Left atrial dilation. '},
    'LVD':{'0': 'No left ventricular dilation. ', '1': 'Left ventricular dilation. '},
    'RAD':{'0': 'No right atrial dilation. ', '1': 'Right atrial dilation. '},
    'RVD':{'0': 'No right ventricular dilation. ', '1': 'Right ventricular dilation. '},
    'PE':{'0': 'No pericardial effusion. ', '1': 'Pericardial effusion. '},
    'LVH':{'0': 'No left ventricular hypertrophy. ', '1': 'Left ventricular hypertrophy. '},
    'IMT':{'0': 'No intracardiac masses and thrombi / singular thrombus. ', '1': 'Intracardiac masses and thrombi / singular thrombus. '},
    'IS':{'0': 'No intracavitary shunts. ', '1': 'Intracavitary shunts. '},
    'AV_regurgitation':{'0': 'No aortic valve regurgitation. ', '1': 'Aortic valve regurgitation. '},
    'AV_stenosis':{'0': 'No aortic valve stenosis. ', '1': 'Aortic valve stenosis. '},
    'AV_vegetations':{'0': 'No aortic valve vegetations. ', '1': 'Aortic valve vegetations. '},
    'MV_regurgitation':{'0': 'No mitral valve regurgitation. ', '1': 'Mitral valve regurgitation. '},
    'MV_stenosis':{'0': 'No mitral valve stenosis. ', '1': 'Mitral valve stenosis. '},
    'MV_vegetations':{'0': 'No mitral valve vegetations. ', '1': 'Mitral valve vegetations. '},
    'PV_regurgitation':{'0': 'No pulmonary valve regurgitation. ', '1': 'Pulmonary valve regurgitation. '},
    'PV_stenosis':{'0': 'No pulmonary valve stenosis. ', '1': 'Pulmonary valve stenosis. '},
    'PV_vegetations':{'0': 'No pulmonary valve vegetations. ', '1': 'Pulmonary valve vegetations. '},
    'TV_regurgitation':{'0': 'No tricuspid valve regurgitation. ', '1': 'Tricuspid valve regurgitation. '},
    'TV_stenosis':{'0': 'No tricuspid valve stenosis. ', '1': 'Tricuspid valve stenosis. '},
    'TV_vegetations':{'0': 'No tricuspid valve vegetations. ', '1': 'Tricuspid valve vegetations. '},
    'LHF_severity':{'normal': 'Normal left heart function. ', 'mild': 'Mild left heart function. ', 'moderate': 'Moderate left heart function. ', 'severe': 'Severe left heart function. '},
    'RHF_severity':{'normal': 'Normal right heart function. ', 'mild': 'Mild right heart function. ', 'moderate': 'Moderate right heart function. ', 'severe': 'Severe right heart function. '},
    'DF_severity':{'normal': 'Normal diastolic function. ', 'grade 1': 'Grade 1 diastolic dysfunction. ', 'grade 2': 'Grade 2 diastolic dysfunction. ', 'grade 3': 'Grade 3 diastolic dysfunction. '},
    'LAD_severity':{'normal': 'Normal left atrial size. ', 'mild': 'Mild left atrial dilation. ', 'moderate': 'Moderate left atrial dilation. ', 'severe': 'Severe left atrial dilation. '},
    'LVD_severity':{'normal': 'Normal left ventricular size. ', 'mild': 'Mild left ventricular dilation. ', 'moderate': 'Moderate left ventricular dilation. ', 'severe': 'Severe left ventricular dilation. '},
    'RAD_severity':{'normal': 'Normal right atrial size. ', 'mild': 'Mild right atrial dilation. ', 'moderate': 'Moderate right atrial dilation. ', 'severe': 'Severe right atrial dilation. '},
    'RVD_severity':{'normal': 'Normal right ventricular size. ', 'mild': 'Mild right ventricular dilation. ', 'moderate': 'Moderate right ventricular dilation. ', 'severe': 'Severe right ventricular dilation. '},
    'PE_severity':{'small': 'Small pericardial effusion. ', 'moderate': 'Moderate pericardial effusion. ', 'large': 'Large pericardial effusion. ', 'tamponade physiology': 'Tamponade physiology. '},
    'LVH_severity':{'normal': 'Normal left ventricular wall thickness. ', 'mild': 'Mild left ventricular hypertrophy. ', 'moderate': 'Moderate left ventricular hypertrophy. ', 'severe': 'Severe left ventricular hypertrophy. '},
    'AV_regurgitation_severity':{'normal': 'Normal aortic valve. ', 'mild': 'Mild aortic regurgitation. ', 'moderate': 'Moderate aortic regurgitation. ', 'severe': 'Severe aortic regurgitation. '},
    'AV_stenosis_severity':{'normal': 'Normal aortic valve. ', 'mild': 'Mild aortic stenosis. ', 'moderate': 'Moderate aortic stenosis. ', 'severe': 'Severe aortic stenosis. '},
    'MV_regurgitation_severity':{'normal': 'Normal mitral valve. ', 'mild': 'Mild mitral regurgitation. ', 'moderate': 'Moderate mitral regurgitation. ', 'severe': 'Severe mitral regurgitation. '},
    'MV_stenosis_severity':{'normal': 'Normal mitral valve. ', 'mild': 'Mild mitral stenosis. ', 'moderate': 'Moderate mitral stenosis. ', 'severe': 'Severe mitral stenosis. '},
    'PV_regurgitation_severity':{'normal': 'Normal pulmonary valve. ', 'mild': 'Mild pulmonary regurgitation. ', 'moderate': 'Moderate pulmonary regurgitation. ', 'severe': 'Severe pulmonary regurgitation. '},
    'PV_stenosis_severity':{'normal': 'Normal pulmonary valve. ', 'mild': 'Mild pulmonary stenosis. ', 'moderate': 'Moderate pulmonary stenosis. ', 'severe': 'Severe pulmonary stenosis. '},
    'TV_regurgitation_severity':{'normal': 'Normal tricuspid valve. ', 'mild': 'Mild tricuspid regurgitation. ', 'moderate': 'Moderate tricuspid regurgitation. ', 'severe': 'Severe tricuspid regurgitation. '},
    'TV_stenosis_severity':{'normal': 'Normal tricuspid valve. ', 'mild': 'Mild tricuspid stenosis. ', 'moderate': 'Moderate tricuspid stenosis. ', 'severe': 'Severe tricuspid stenosis. '},
    'TAPSE_severity':{'normal': 'Normal TAPSE. ', 'mild': 'Mild TAPSE. ', 'moderate': 'Moderate TAPSE. ', 'severe': 'Severe TAPSE. '}
}



def argparse_args():
    """
    Parse the arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Start and end index of the studies to process")
    parser.add_argument("--start", type=int, default=None, help="Start index of the studies to process")
    parser.add_argument("--end", type=int, default=None, help="End index of the studies to process")
    parser.add_argument("--step", type=int, default=1, help="Step size of the studies to process")
    parser.add_argument("--csv_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/", help="Path to the csv directory")
    parser.add_argument("--dataset_dir", type=str, default="/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/", help="Path to the dataset directory")
    parser.add_argument("--raw_dataset_dir", type=str, default="/mnt/hanoverdev/data/hanwen/ECHO/deidentified/", help="Path to the report directory")
    parser.add_argument("--studies_with_report_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_report.json", help="Path to the studies_with_report json file")
    parser.add_argument("--studies_with_labels_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_label.json", help="Path to the studies_with_label json file")
    parser.add_argument("--labels_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_labels.json", help="Path to the labels json file")
    parser.add_argument("--video_embedding_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/video_embedding_v4/", help="Path to the video embedding file")
    parser.add_argument("--report_embedding_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/report_embedding_v4/", help="Path to the report embedding file")
    parser.add_argument("--merged_video_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/merged_video_embedding_v4.pt", help="Path to the output merged video embedding file")
    parser.add_argument("--merged_report_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/merged_report_embedding_v4.pt", help="Path to the output merged report embedding file")
    parser.add_argument("--best_videos_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/best_video.json", help="Path to the best video json file")
    parser.add_argument("--new_csv_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_dataset_v4/", help="Path to the new dataset csv directory")
    
    parser.add_argument("--save_studies", action="store_true", default=False, help="Save the studies that need to compute embeddings")
    parser.add_argument("--get_labels", action="store_true", default=False, help="Get the labels of the studies")
    parser.add_argument("--compute_video_embeddings", action="store_true", default=False, help="Compute the embeddings for every video in the studies in the pt file")
    parser.add_argument("--compute_report_embeddings", action="store_true", default=False, help="Compute the embeddings for the report in the studies in the pt file")
    parser.add_argument("--check_progress", action="store_true", default=False, help="check_progress the progress of the studies")
    parser.add_argument("--merge_video_embeddings", action="store_true", default=False, help="Merge the embeddings into one file")
    parser.add_argument("--merge_report_embeddings", action="store_true", default=False, help="Merge the embeddings into one file")
    parser.add_argument("--get_best_videos", action="store_true", default=False, help="Filter the dataset to only include the videos that have largest similarity with prompts in each studies, output into a csv file")
    parser.add_argument("--split", action="store_true", default=False, help="Split the dataset into train, val and test set")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for computation")
    return parser.parse_args()


def load_model(model_name, device):
    """
    Load the model.
    """
    print("✊✊✊ Start loading model ✊✊✊")
    model, _, _ = create_model_and_transforms(model_name, precision='bf16', device=device)
    print("🎉🎉🎉 Finish loading model 🎉🎉🎉")
    return model

def get_all_studies_with_labels(task_set, csv_dir, **kwargs):
    """
    Get all studies that exists in task_set.
    """
    studies = set()
    for task in task_set:
        print(f"processing {task}😊😊😊")
        df = pd.read_csv(os.path.join(csv_dir, task + '.csv'), dtype=str)
        studies.update(list(df['path'].unique()))
    return sorted(list(studies))

def not_regular_report(path, exception_words):
    """
    Check if the report is not a regular report.
    """
    with open(path, "r") as f:
        lines = f.readlines()
        for i in range(min(len(lines), 5)):
            if "Codes" in lines[i]:
                for word in exception_words:
                    if word in lines[i + 1]:
                        return True
    return False

def get_studies_with_regular_report(studies, raw_dataset_dir, exception_words = ["Stress"], **kwargs):
    """
    Get studies with regular report.
    """
    original_size_dir = "/mnt/hanoverdev/data/hanwen/ECHO/original_size/"
    
    assert os.path.exists(original_size_dir), f"Directory {raw_dataset_dir} does not exist"
    
    regular_studies = []
    for study in tqdm(studies, desc="Filtering regular reports", total=len(studies)):
        study_path = os.path.join(raw_dataset_dir, study)
        txt_files = [f for f in os.listdir(study_path) if f.endswith('.txt')]
        if len(txt_files) == 0:
            continue
        txt_file = sorted(txt_files)[-1]  # Get the last file
        if not_regular_report(os.path.join(raw_dataset_dir, study, txt_file), exception_words) == False:
            regular_studies.append(study)

    return regular_studies

def get_study_labels_dict(task_set, studies, csv_dir, **kwargs):
    """
    Get the labels of studies in task_set.
    """
    template = {task: None for task in task_set}
    studies_labels = {study: template.copy() for study in studies}
    
    print("✊✊✊ Start getting labels of studies... ✊✊✊")
    for task in task_set:
        df = pd.read_csv(os.path.join(csv_dir, task + '.csv'), dtype=str)
        for label, study in zip(list(df['label']), list(df['path'])):
            if study in studies_labels:
                studies_labels[study][task] = label
    print("🎉🎉🎉 Finished getting labels of studies. 🎉🎉🎉")
    return studies_labels

def compute_video_embeddings(model, studies, dataset_dir, video_embedding_dir, device, **kwargs):
    """
    Compute the video embeddings for the studies, and save embeddings of videos in each study in a separate file in video_embedding_dir.
    """
    mean_tensor = torch.tensor(model.visual.image_mean or (0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(model.visual.image_std or (0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1).to(device)
    
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
                test_video = F.interpolate(test_video.to(torch.float32), size=model.visual.image_size, mode="bicubic", align_corners=False) / 255.0
                test_video = (test_video - mean_tensor) / std_tensor
                test_video = test_video.to(torch.bfloat16)
                
                test_video_embedding = F.normalize(model.encode_image(test_video), dim=-1)
                embeddings[short_path] = test_video_embedding.mean(dim=0)
                
                pbar.update(1)
                
        torch.save(embeddings, os.path.join(video_embedding_dir, f"{study}.pt"))
        print(f"Embeddings saved to {os.path.join(video_embedding_dir, f'{study}.pt')}")
    print("🎉🎉🎉 All video embeddings saved! 🎉🎉🎉")

def generate_report(studies_labels, filtering_tasks_set):
    """
    Generate the report for the studies with labels.
    """
    assert set(filtering_tasks_set) <= set(prompts.keys()), f"Filtering tasks {filtering_tasks_set} not in prompts"
    reports = dict()
    for study, labels_of_study in studies_labels.items():
        prompt = ""
        for task in filtering_tasks_set:
            assert task in prompts, f"Task {task} not in prompts"
            if labels_of_study[task] is not None:
                assert labels_of_study[task] in labels[task], f"Label {labels_of_study[task]} not in {labels[task]}"
                prompt += prompts[task][labels_of_study[task]]
        reports[study] = prompt
    return reports

def compute_report_embeddings(model, reports, report_embedding_dir, device, **kwargs):
    """
    Compute the report embeddings for the studies, and save each embedding in a file.
    
    Args:
        reports (dict): Dictionary of reports with study names as keys and report text as values.
    """
    if not os.path.exists(report_embedding_dir):
        os.makedirs(report_embedding_dir)
    
    for study, prompt in tqdm(reports.items(), desc="Computing Report Embeddings", total=len(reports)):
        report_embedding_path = os.path.join(report_embedding_dir, f"{study}.pt")
        if os.path.exists(report_embedding_path):
            print(f"Report embedding for {study} already exists, skipping...")
            continue
        
        text = tokenize(prompt.upper()).to(device)
        report_embedding = F.normalize(model.encode_text(text), dim=-1)
        
        torch.save(report_embedding, report_embedding_path)
        
    print("🎉🎉🎉 All report embeddings saved! 🎉🎉🎉")

def merge_embeddings(embedding_dir, merged_embedding_path, **kwargs):
    """
    Merge all the embeddings in the embedding_dir into one file.
    """
    merged_embeddings = dict()
    
    for file in tqdm(os.listdir(embedding_dir), desc=f"Merging Embeddings in {embedding_dir.split('/')[-1]}", total=len(os.listdir(embedding_dir))):
        if file.endswith('.pt'):
            study = file[:-3]
            embedding = torch.load(os.path.join(embedding_dir, file))
            merged_embeddings[study] = embedding
    
    torch.save(merged_embeddings, merged_embedding_path)
    print(f"🎉🎉🎉 Merged embeddings saved to {merged_embedding_path} 🎉🎉🎉")
    return merged_embeddings

def check_progress(studies, video_embedding_dir, report_embedding_dir, **kwargs):
    """
    Check the progress of the studies.
    """
    if not os.path.exists(video_embedding_dir):
        print(f"Directory {video_embedding_dir} does not exist")
    
    if not os.path.exists(report_embedding_dir):
        print(f"File {report_embedding_dir} does not exist")
    
    video_processed = False
    report_processed = False
    video_processed_intervals = []
    report_processed_intervals = []
    video_start = 0
    report_start = 0
    video_cnt = 0
    report_cnt = 0
    
    for idx, study in enumerate(tqdm(studies, desc="Checking Progress", total=len(studies))):
        if os.path.exists(os.path.join(video_embedding_dir, f"{study}.pt")) and not video_processed:
            video_processed = True
            video_start = idx
        elif not os.path.exists(os.path.join(video_embedding_dir, f"{study}.pt")) and video_processed:
            video_processed = False
            video_processed_intervals.append((video_start, idx - 1))
            video_cnt += idx - video_start
        
        if os.path.exists(os.path.join(report_embedding_dir, f"{study}.pt")) and not report_processed:
            report_processed = True
            report_start = idx
        elif not os.path.exists(os.path.join(report_embedding_dir, f"{study}.pt")) and report_processed:
            report_processed = False
            report_processed_intervals.append((report_start, idx - 1))
            report_cnt += idx - report_start
    if video_processed:
        video_processed_intervals.append((video_start, len(studies) - 1))
        video_cnt += len(studies) - video_start
    if report_processed:
        report_processed_intervals.append((report_start, len(studies) - 1))
        report_cnt += len(studies) - report_start
    
    print(f"Video processed intervals: {video_processed_intervals}")
    print(f"Report processed intervals: {report_processed_intervals}")
    print(f"Video processed {video_cnt} / {len(studies)}")
    print(f"Report processed {report_cnt} / {len(studies)}")

def get_best_videos(studies, video_embeddings, prompt_embeddings, device, **kwargs):
    best_videos = dict()
    max_similarity_sum = 0
    max_similarity_count = 0
    all_similarity_sum = 0
    all_similarity_count = 0
    with tqdm(total=len(studies), desc="Getting best videos") as pbar:
        for study in studies:
            max_similarity = 0
            max_video = None
            
            prompt_embedding = prompt_embeddings[study].to(device)
            for video_path, video_embedding in video_embeddings[study].items():
                video_embedding = video_embedding.to(device)
                similarity = (video_embedding @ prompt_embedding.T).item()
                assert similarity >= 0, f"Similarity {similarity} is negative"
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_video = video_path
                all_similarity_sum += similarity
                all_similarity_count += 1
            
            assert max_video is not None, f"Max video is None for study {study}"
            best_videos[study] = max_video
            max_similarity_sum += max_similarity
            max_similarity_count += 1
            pbar.update(1)
            
            pbar.set_postfix({
                "avg max_similarity": f"{max_similarity_sum / max_similarity_count:.4f}" if max_similarity_count > 0 else "N/A",
                "similarity_count": max_similarity_count,
                "avg all_similarity": f"{all_similarity_sum / all_similarity_count:.4f}" if all_similarity_count > 0 else "N/A",
                "all_similarity_count": all_similarity_count,
            })
            
    return best_videos

def split_dataset(studies):
    return
    