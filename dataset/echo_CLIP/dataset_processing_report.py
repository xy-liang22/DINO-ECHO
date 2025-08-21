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
from utils import load_model, argparse_args, get_all_studies_with_labels, get_studies_with_regular_report, get_study_labels_dict, compute_video_embeddings, generate_report, compute_report_embeddings, merge_embeddings, check_progress, get_best_videos    

task_set = ['AV_regurgitation_severity', 'AV_regurgitation', 'AV_stenosis_severity', 'AV_stenosis', 'AV_vegetations', 'DF_severity', 'DF', 'IMT', 'IS', 'LAD_severity', 'LAD', 'LHF_severity', 'LHF', 'LVD_severity', 'LVD', 'LVH_severity', 'LVH', 'MV_regurgitation_severity', 'MV_regurgitation', 'MV_stenosis_severity', 'MV_stenosis', 'MV_vegetations', 'PE_severity', 'PE', 'PV_regurgitation_severity', 'PV_regurgitation', 'PV_stenosis_severity', 'PV_stenosis', 'PV_vegetations', 'RAD_severity', 'RAD', 'RHF_severity', 'RHF', 'RVD_severity', 'RVD', 'TAPSE_severity', 'TV_regurgitation_severity', 'TV_regurgitation', 'TV_stenosis_severity', 'TV_stenosis', 'TV_vegetations']
# task_set = ['LHF']

filtering_tasks = ['AV_regurgitation', 'AV_stenosis', 'AV_vegetations', 'DF', 'IMT', 'IS', 'LAD', 'LHF', 'LVD', 'LVH', 'MV_regurgitation', 'MV_stenosis', 'MV_vegetations', 'PE',  'PV_regurgitation', 'PV_stenosis', 'PV_vegetations', 'RAD', 'RHF', 'RVD', 'TAPSE_severity',  'TV_regurgitation', 'TV_stenosis', 'TV_vegetations']


args = argparse_args()
args.device = torch.device(args.device)

echo_clip = None
if args.compute_video_embeddings or args.compute_report_embeddings or args.get_best_videos:
    echo_clip = load_model("hf-hub:mkaichristensen/echo-clip", device=args.device)

studies = None
studies_with_report = None
if args.save_studies:
    studies = get_all_studies_with_labels(task_set, args.csv_dir)
    with open(args.studies_with_labels_path, "w") as f:
        json.dump(studies, f, indent=4)
        print(f"Saved {len(studies)} studies to {args.studies_with_labels_path}")
        
    studies_with_report = get_studies_with_regular_report(studies, args.raw_dataset_dir)
    
    with open(args.studies_with_report_path, "w") as f:
        json.dump(studies_with_report, f, indent=4)
        print(f"Saved {len(studies_with_report)} studies to {args.studies_with_report_path}")

if studies is None:
    print(f"Loading studies from {args.studies_with_labels_path}")
    with open(args.studies_with_labels_path, "r") as f:
        studies = json.load(f)
if studies_with_report is None:
    print(f"Loading studies with report from {args.studies_with_report_path}")
    with open(args.studies_with_report_path, "r") as f:
        studies_with_report = json.load(f)

studies_labels = None
if args.get_labels:
    studies_labels = get_study_labels_dict(task_set, studies, args.csv_dir)
    with open(args.labels_path, "w") as f:
        json.dump(studies_labels, f, indent=4)
        print(f"Saved {len(studies_labels)} studies to {args.labels_path}")

if studies_labels is None and args.compute_report_embeddings:
    print(f"Loading studies labels from {args.labels_path}")
    with open(args.labels_path, "r") as f:
        studies_labels = json.load(f)
    
if args.check_progress:
    check_progress(studies, **vars(args))

if args.start is not None and args.end is not None:
    studies = studies[args.start: args.end: args.step]
    if studies_labels is not None:
        studies_labels = {k: studies_labels[k] for k in list(studies_labels.keys())[args.start:args.end]}
    
if args.compute_video_embeddings:
    compute_video_embeddings(echo_clip, studies, **vars(args))

if args.compute_report_embeddings:
    reports = generate_report(studies_labels, filtering_tasks)
    compute_report_embeddings(echo_clip, reports, **vars(args))

video_embeddings = None
if args.merge_video_embeddings:
    video_embeddings = merge_embeddings(args.video_embedding_dir, args.merged_video_embedding_path)
report_embeddings = None
if args.merge_report_embeddings:
    report_embeddings = merge_embeddings(args.report_embedding_dir, args.merged_report_embedding_path)


if args.get_best_videos:
    if video_embeddings is None:
        video_embeddings = torch.load(args.merged_video_embedding_path)
    if report_embeddings is None:
        report_embeddings = torch.load(args.merged_report_embedding_path)
    
    best_videos = get_best_videos(studies, video_embeddings, report_embeddings, args.device)
    
    with open(args.best_videos_path, "w") as f:
        json.dump(best_videos, f, indent=4)
        print(f"🎉🎉🎉 Saved {len(best_videos)} studies to {args.best_videos_path} 🎉🎉🎉")