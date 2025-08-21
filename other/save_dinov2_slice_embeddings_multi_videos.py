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
import monai.transforms as monai_transforms
from custom_util.eval_utils import create_transforms

from tqdm import tqdm
import argparse

def get_all_videos(data_dir, studies_path, list_output_path, dict_output_path, **kwargs):
    
    print(f"🤔🤔🤔 Start loading videos from {data_dir} and {studies_path} ✊✊✊")
    if os.path.exists(list_output_path) and os.path.exists(dict_output_path):
        print(f"List and dict files already exist at {list_output_path} and {dict_output_path}.")
        return
        with open(list_output_path, 'r') as f:
            videos_list = json.load(f)
        with open(dict_output_path, 'r') as f:
            videos_dict = json.load(f)
        return videos_list, videos_dict
    
    assert os.path.exists(data_dir), f"Path {data_dir} does not exist"
    assert os.path.exists(studies_path), f"Path {studies_path} does not exist"    
    # Load the studies from the JSON file
    with open(studies_path, 'r') as f:
        studies = json.load(f)
    
    # Extract the video paths and labels
    videos_list = []
    videos_dict = {}
    with tqdm(total=len(studies), desc="Loading studies", unit="study") as pbar:
        for i, study in enumerate(studies):
            prefix = os.path.join(data_dir, study)
            assert os.path.exists(prefix), f"Path {prefix} does not exist"
            videos = [os.path.join(study, video) for video in os.listdir(prefix) if video.endswith('.npy')]
            videos_dict[study] = videos
            videos_list.extend(videos)
            pbar.update(1)
            
            pbar.set_postfix({"avg_num_videos": len(videos_list) / (i + 1), "study": study})
    
    # Save the videos list to a json file
    with open(list_output_path, 'w') as f:
        json.dump(videos_list, f, indent=4)
        print(f"Saved {len(videos_list)} videos to {list_output_path}, len: {len(videos_list)}")
    # Save the videos dict to a json file
    with open(dict_output_path, 'w') as f:
        json.dump(videos_dict, f, indent=4)
        print(f"Saved {len(videos_dict)} videos to {dict_output_path}")
    # return videos_list, videos_dict
    print(f"🎉🎉🎉 Finished loading videos from {data_dir} and {studies_path} 😊😊😊")

class EchoDatasetMultiVideos(Dataset):
    def __init__(self, 
                 data_dir, 
                 list_output_path,
                 start_idx=None,
                 end_idx=None,
                 embedding_dir=None,
                 image_size=256,
                 frames_ratio=0.5,
                 max_frames=128,
                 **kwargs):
        print(f"🤔🤔🤔 Start to init EchoDatasetMultiVideos ✊✊✊")
        with open(list_output_path, 'r') as f:
            self.videos_list = json.load(f)
            if start_idx is not None and end_idx is not None:
                self.videos_list = self.videos_list[start_idx:end_idx]
            print(f"Start index: {start_idx}, End index: {end_idx}, Total videos: {len(self.videos_list)}")
            self.videos_list = [video for video in tqdm(self.videos_list) if not os.path.exists(os.path.join(embedding_dir, video.replace(".npy", ".pt")))]
        self.data_dir = data_dir
        self.processor = monai_transforms.Resized(keys=["pixel_values"], spatial_size=(-1, image_size, image_size), mode="bilinear")
        self.frames_ratio = frames_ratio
        self.max_frames = max_frames
        print(f"🎉🎉🎉 Finished init EchoDataset, total {len(self.videos_list)} videos ✊✊✊")
    
    def __len__(self):
        return len(self.videos_list)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.videos_list[idx])
        video = np.load(video_path)
        video_tensor = torch.tensor(video).float()
        if self.processor is not None:
            video_tensor = self.processor({"pixel_values": video_tensor})["pixel_values"]
        if video_tensor.shape[1] > self.max_frames:
            video_tensor = monai_transforms.Resized(
                keys=["pixel_values"], 
                spatial_size=(video_tensor.shape[1] * self.frames_ratio, -1, -1)
            )({"pixel_values": video_tensor})["pixel_values"]
        if video_tensor.shape[1] > self.max_frames:
            video_tensor = video_tensor[:, :self.max_frames, :, :]
        study = self.videos_list[idx].split("/")[0]
        return {
            'video': video_tensor,
            'study': study,
            'path': self.videos_list[idx]
        }

def collate_fn_padding(batch):
    """Custom collate function to handle variable length video tensors and return with a mask."""
    max_length = max(data['video'].shape[1] for data in batch)
    channels = batch[0]['video'].shape[0]
    # Pad videos to the maximum length in the batch
    padded_videos = []
    masks = []
    studies = []
    paths = []
    for data in batch:
        video = data['video']
        padded_tensor = torch.zeros((channels, max_length, *video.shape[2:]), dtype=video.dtype)
        padded_tensor[:, :video.shape[1]] = video
        padded_videos.append(padded_tensor)
        masks.append(torch.tensor([1] * video.shape[1] + [0] * (max_length - video.shape[1]), dtype=torch.bool))
        studies.append(data['study'])
        paths.append(data['path'])
    padded_videos = torch.stack(padded_videos, dim=0)  # Shape: (B, C, T, W, H)
    masks = torch.stack(masks, dim=0)  # Shape: (B, max_length)
    assert len(padded_videos.shape) == 5, f"Expected padded_videos to have 5 dimensions, got {len(padded_videos.shape)}"
    return (padded_videos, masks), studies, paths

class EmbeddingsDataset(Dataset):
    def __init__(self,
                 embedding_dir,
                 embedding_paths):
        self.embedding_paths = [os.path.join(embedding_dir, path.replace(".npy", ".pt")) for path in embedding_paths]
    
    def __len__(self):
        return len(self.embedding_paths)
    
    def __getitem__(self, idx):
        embedding = torch.load(self.embedding_paths[idx])
        return embedding

class DINOv2HiddenState(nn.Module):
    def __init__(self, 
                 config_path: str,
                 pretrained: str=None,
                 **kwargs):
        super(DINOv2HiddenState, self).__init__()
        self.vit, _ = make_dinov2(config_path, ckpt_path=pretrained)
        self.transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x, masks = x
        assert len(x.shape) == 5
        B, C, F, H, W = x.shape

        # reshape input [B, C, F, H, W] -> [B * F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # normalize the input
        if x.max() > 1:
            x = x / 255.0
        
        x = self.transform(x)
        
        # run through the backbone
        embedding_output = self.vit(x) # [B * F, hidden_dim]
        embedding_output = embedding_output.view(B, F, -1) # [B, F, hidden_dim]
        # embedding_output = embedding_output * masks.unsqueeze(-1)
        # # logits = embedding_output.sum(dim=1) / masks.sum(dim=1, keepdim=True)  # [B, hidden_dim]
        # logits = embedding_output.sum(dim=1)  # [B, hidden_dim]

        if not isinstance(embedding_output, torch.Tensor):
            embedding_output = embedding_output.as_tensor()
        return embedding_output, masks.sum(dim=1)  # return logits and the sum of masks for each video

def parse_args():
    parser = argparse.ArgumentParser(description="Save embeddings for ECHO dataset")
    parser.add_argument("--data_dir", type=str, default="/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/", help="Path to the ECHO dataset")
    parser.add_argument("--studies_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/studies_with_label.json", help="Path to the studies with labels JSON file")
    parser.add_argument("--list_output_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos.json", help="Path to save the list of videos")
    parser.add_argument("--dict_output_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/all_videos_dict.json", help="Path to save the dictionary of videos")
    parser.add_argument("--embedding_dir", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/dinov2_clip_embeddings/", help="Directory to save the embeddings")
    parser.add_argument("--combined_embedding_path", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/dinov2_clip_embeddings_mean.pt", help="Path to save the combined embeddings")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for the videos list")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for the videos list")
    parser.add_argument("--pretrained", type=str, default="/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_clip.pt", help="Path to the pretrained DINOv2 model")
    parser.add_argument("--config_path", type=str, default="models/dinov2_modules/configs/train/vitl16_lbsz48_short.yaml", help="Path to the DINOv2 config file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing videos")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for processing (cpu or cuda)")
    parser.add_argument("--save_embeddings", action="store_true", help="Whether to save the embeddings or not")
    parser.add_argument("--combine_embeddings", action="store_true", help="Whether to combine the embeddings or not")
    parser.add_argument("--mean_embeddings", action="store_true", help="Whether to compute the mean embeddings or not")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for the DINOv2 model")
    parser.add_argument("--frames_ratio", type=float, default=0.5, help="Ratio of frames to use from each video")
    parser.add_argument("--max_frames", type=int, default=128, help="Maximum number of frames to use from each video")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.save_embeddings:
        get_all_videos(**vars(args))
        dataset = EchoDatasetMultiVideos(**vars(args))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, collate_fn=collate_fn_padding)
        model = DINOv2HiddenState(**vars(args))
        model.to(args.device)
        with tqdm(total=len(dataloader), desc="Processing videos", unit="batch") as pbar:
            for i, (video_with_mask, study, video_path) in enumerate(dataloader):
                if isinstance(video_with_mask, (tuple, list)):
                    video_with_mask[0] = video_with_mask[0].to(args.device)
                    video_with_mask[1] = video_with_mask[1].to(args.device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        embedding, num_frames = model(video_with_mask)
                for j in range(embedding.shape[0]):
                    video_path_ = video_path[j]
                    study_ = study[j]
                    if not os.path.exists(os.path.join(args.embedding_dir, study_)):
                        os.makedirs(os.path.join(args.embedding_dir, study_), exist_ok=True)
                    embedding_path = os.path.join(args.embedding_dir, video_path_.replace(".npy", ".pt"))
                    # torch.save({"embedding": embedding[j, :num_frames[j]].cpu(), "num_frames": num_frames[j].cpu()}, embedding_path)
                    torch.save(embedding[j, :num_frames[j]].cpu(), embedding_path)
                pbar.update(1)
                if args.start_idx is not None:
                    pbar.set_postfix({"start_idx": args.start_idx, "cnt_idx": args.start_idx + i, "study": study_})
                else:
                    pbar.set_postfix({"cnt_idx": i, "study": study_})
    if args.combine_embeddings:
        print(f"🤔🤔🤔 Start to combine embeddings ✊✊✊")
        with open(args.dict_output_path, 'r') as f:
            videos_dict = json.load(f)
        all_embeddings = {}
        studies = [study for study in videos_dict.keys() if not os.path.exists(os.path.join(args.embedding_dir, f"{study}.pt"))]
        print(f"Total studies: {len(studies)} Avg number of videos per study: {sum(len(videos_dict[study]) for study in studies) / len(studies)}")
        for study in tqdm(studies, desc="Loading embeddings", unit="study"):
            videos = videos_dict[study]
            # embeddings_this_study = {video.replace(".npy", ".pt").split('/')[-1]: torch.load(os.path.join(args.embedding_dir, video.replace(".npy", ".pt"))) for video in videos}
            embeddings_this_study = torch.cat([torch.load(os.path.join(args.embedding_dir, video.replace(".npy", ".pt"))) for video in videos])
            if embeddings_this_study.shape[0] > 1024:
                import math
                embeddings_this_study = embeddings_this_study[:: (math.ceil(embeddings_this_study.shape[0] / 1024))]
            torch.save(embeddings_this_study, os.path.join(args.embedding_dir, f"{study}.pt"))
        print(f"🎉🎉🎉 Finished combining embeddings, saved to {args.embedding_dir} ✊✊✊")

if __name__ == "__main__":
    main()