import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dinov2_as_classifier import make_dinov2
from custom_util.eval_utils import create_transforms

from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Save embeddings for ECHO dataset")
    parser.add_argument("--studies_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_label.json", help="Path to the studies with labels JSON file")
    parser.add_argument("--list_output_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos.json", help="Path to save the list of videos")
    parser.add_argument("--dict_output_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos_dict.json", help="Path to save the dictionary of videos")
    parser.add_argument("--embedding_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/video_embedding_v4/", help="Directory to save the embeddings")
    parser.add_argument("--combined_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/echoclip_clip_embeddings_mean.pt", help="Path to save the combined embeddings")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for the videos list")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for the videos list")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing videos")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for processing (cpu or cuda)")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🤔🤔🤔 Start to compute mean embeddings ✊✊✊")
    with open(args.studies_path, 'r') as f:
        studies = json.load(f)
    all_embeddings = {}
    for study in tqdm(studies, desc="Computing mean embeddings", unit="study"):
        embeddings_this_study_dict = torch.load(os.path.join(args.embedding_dir, f"{study}.pt"))
        all_embeddings[study] = torch.mean(torch.stack(list(embeddings_this_study_dict.values()), dim=0), dim=0).cpu()
        # embeddings_dataset = EmbeddingsDataset(args.embedding_dir, videos)
        # dataloader = torch.utils.data.DataLoader(embeddings_dataset, batch_size=len(embeddings_dataset), num_workers=20, shuffle=False)
        # times = 0
        # for batch in dataloader:
        #     times += 1
        #     embeddings_this_study = batch
        # assert times == 1, f"Expected only one batch, but got {times} batches for study {study}"
        # all_embeddings[study] = torch.mean(embeddings_this_study, dim=0)
    torch.save(all_embeddings, args.combined_embedding_path)
    print(f"🎉🎉🎉 Finished combining embeddings, saved to {args.combined_embedding_path} ✊✊✊")

if __name__ == "__main__":
    main()