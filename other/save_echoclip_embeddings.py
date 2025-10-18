import os
import json
import torch
from torch import nn
from torchvision import transforms

from save_biomedclip_embeddings import parse_args, get_all_videos, EchoDataset, BiomedCLIP
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_util.eval_utils import create_transforms
from open_clip import create_model_and_transforms

from tqdm import tqdm

class EchoCLIP(nn.Module):
    def __init__(self, device, **kwargs):
        super(EchoCLIP, self).__init__()
        model, _, _ = create_model_and_transforms("hf-hub:mkaichristensen/echo-clip", precision="bf16", device=device)
        print(model.visual.preprocess_cfg)
        self.vit = model.visual
        self.transform = transforms.Normalize(mean=model.visual.image_mean or (0.48145466, 0.4578275, 0.40821073), std=model.visual.image_std or (0.26862954, 0.26130258, 0.27577711))
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