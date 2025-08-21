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
from open_clip import create_model_and_transforms

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

class EchoDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 list_output_path,
                 start_idx=None,
                 end_idx=None,
                 embedding_dir=None,
                 image_size=256,
                 **kwargs):
        print(f"🤔🤔🤔 Start to init EchoDataset ✊✊✊")
        with open(list_output_path, 'r') as f:
            self.videos_list = json.load(f)
            if start_idx is not None and end_idx is not None:
                self.videos_list = self.videos_list[start_idx:end_idx]
            print(f"Start index: {start_idx}, End index: {end_idx}, Total videos: {len(self.videos_list)}")
            self.videos_list = [video for video in tqdm(self.videos_list) if not os.path.exists(os.path.join(embedding_dir, video.replace(".npy", ".pt")))]
        self.data_dir = data_dir
        self.processor, _ = create_transforms(image_size=image_size, num_frames=64, max_frames=128, dataclass="EchoData")
        print(f"🎉🎉🎉 Finished init EchoDataset, total {len(self.videos_list)} videos ✊✊✊")
    
    def __len__(self):
        return len(self.videos_list)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.videos_list[idx])
        video = np.load(video_path)
        video_tensor = torch.tensor(video).float()
        video_tensor = self.processor({"pixel_values": video_tensor})["pixel_values"]
        study = self.videos_list[idx].split("/")[0]
        return video_tensor, study, self.videos_list[idx]

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

class BiomedCLIP(nn.Module):
    def __init__(self, **kwargs):
        super(BiomedCLIP, self).__init__()
        model, _, _ = create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        print(model.visual.preprocess_cfg)
        self.vit = model.visual
        self.transform = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    def forward(self, x):
        assert len(x.shape) == 5
        B, C, F, H, W = x.shape
        # reshape input [B, C, F, H, W] -> [B * F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # normalize the input
        if x.max() > 1:
            x = x / 255.0
            
        x = self.transform(x)
        embedding_output = self.vit(x).view(B, F, -1) # [B * F, hidden_dim]
        # mean pooling
        h = torch.mean(embedding_output, dim=1) # [B, hidden_dim]
        return h

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
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.save_embeddings:
        get_all_videos(**vars(args))
        dataset = EchoDataset(**vars(args))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
        model = BiomedCLIP(**vars(args))
        model.to(args.device)
        print("Embedding_dir:", args.embedding_dir)
        with tqdm(total=len(dataloader), desc="Processing videos", unit="batch") as pbar:
            for i, (video, study, video_path) in enumerate(dataloader):
                video = video.to(args.device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        embedding = model(video)
                for j in range(embedding.shape[0]):
                    video_path_ = video_path[j]
                    study_ = study[j]
                    if not os.path.exists(os.path.join(args.embedding_dir, study_)):
                        os.makedirs(os.path.join(args.embedding_dir, study_), exist_ok=True)
                    embedding_path = os.path.join(args.embedding_dir, video_path_.replace(".npy", ".pt"))
                    torch.save(embedding[j].cpu(), embedding_path)
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
        if len(studies):
            print(f"Total studies: {len(studies)} Avg number of videos per study: {sum(len(videos_dict[study]) for study in studies) / len(studies)}")
        for study in tqdm(studies, desc="Loading embeddings", unit="study"):
            videos = videos_dict[study]
            embeddings_this_study = {video.replace(".npy", ".pt").split('/')[-1]: torch.load(os.path.join(args.embedding_dir, video.replace(".npy", ".pt"))) for video in videos}
            torch.save(embeddings_this_study, os.path.join(args.embedding_dir, f"{study}.pt"))
        print(f"🎉🎉🎉 Finished combining embeddings, saved to {args.embedding_dir} ✊✊✊")
    if args.mean_embeddings:
        print(f"🤔🤔🤔 Start to compute mean embeddings ✊✊✊")
        with open(args.studies_path, 'r') as f:
            studies = json.load(f)
        all_embeddings = {}
        for study in tqdm(studies, desc="Computing mean embeddings", unit="study"):
            embeddings_this_study_dict = torch.load(os.path.join(args.embedding_dir, f"{study}.pt"))
            all_embeddings[study] = torch.mean(torch.stack(list(embeddings_this_study_dict.values()), dim=0), dim=0)
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