import os
import sys
OPEN_CLIP_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'open_clip', 'src')
sys.path.insert(0, OPEN_CLIP_SRC_PATH)
from open_clip.model import CLIPVisionCfg, CLIPTextCfg
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
import torch.nn as nn
from typing import Union

class CLIPClassifier(nn.Module):
    def __init__(
        self,
        clip_model_name,
        pretrained: str='',
        device: Union[str, torch.device]='cpu',
        num_classes: int=2,
        logit_scale: float=14.34474,
        proj: bool=False,  # Whether to use projection layer
        **kwargs
    ):
        super().__init__()
        self.clip, _ = create_model_from_pretrained(
            model_name=clip_model_name,
            pretrained=pretrained,
            device=device,
            output_dict=True
        )
        if getattr(self.clip.visual, 'proj', None) is not None and proj:
            self.clip.visual = self.clip.visual.proj
            print(f"Using projection layer from {clip_model_name} visual model.")
        else:
            self.clip.visual = nn.Identity()
            print(f"No projection layer found in {clip_model_name} visual model, using Identity layer instead.")
        self.clip.logit_scale = nn.Parameter(torch.tensor(logit_scale, dtype=torch.float32, device=device))
            
        self.num_classes = num_classes
        # self.device = device
        # self.tokenize = get_tokenizer(clip_model_name)

    def forward(self, input):
        """
        video_embeddings: (B, 1024)
        texts: (B, 1) or (B, T)
        """
        # if isinstance(texts, str):
        #     texts = [texts]
        # elif isinstance(texts, list):
        #     pass
        # else:
        #     raise ValueError("texts should be a string or a list of strings.")
        
        # text_tokens = self.tokenize(texts).to(self.device)
        video_embeddings, text_tokens = input
        # print(f"Video embeddings shape: {video_embeddings.shape}, Text tokens shape: {text_tokens.shape}")
        # B, N, L = text_tokens.shape
        text_tokens = text_tokens[0] # all text tokens are the same, so we can just take the first one
        # print(f"Reshaped text tokens shape: {text_tokens.shape} (B={B}, N={N}, L={L})")
        output = self.clip(
            image=video_embeddings,
            text=text_tokens
        )
        image_features = output['image_features']
        text_features = output['text_features']
        logit_scale = output['logit_scale']
        # print(f"Image features shape: {image_features.shape}, Text features shape: {text_features.shape}, Logit scale: {logit_scale}")
        print(f"Logit scale: {logit_scale}")
        print(f"self.clip.logit_scale: {self.clip.logit_scale}")
        logits = logit_scale * image_features @ text_features.T
        # print((image_features @ text_features.T).abs().mean())
        # print(f"Logits shape: {logits.shape}, Image features shape: {image_features.shape}, Text features shape: {text_features.shape}")
        # print(logits.isnan().to(torch.float32).mean())
        return logits

def clip_classifier(clip_model_name, pretrained='', device='cpu', num_classes=2, logit_scale=2.66, **kwargs):
    """
    Factory function to create a CLIPClassifier instance.
    """
    return CLIPClassifier(
        clip_model_name=clip_model_name,
        pretrained=pretrained,
        device=device,
        num_classes=num_classes,
        logit_scale=logit_scale,
        proj="DINOv2" in clip_model_name,  # Use projection layer for DINOv2 models
        **kwargs
    )

if __name__ == '__main__':
    # print(OPEN_CLIP_SRC_PATH)
    # import open_clip
    # open_clip_path = open_clip.__path__
    # print(open_clip_path)
    model = CLIPClassifier(
        clip_model_name="DINOv2_BiomedBERT_study_new",
        pretrained="/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs/BiomedBERT_study_original1_proj_dist/checkpoints/epoch_20.pt",
        device="cuda:0"
    )
    print(model)
    from open_clip import get_tokenizer
    tokenizer = get_tokenizer("DINOv2_BiomedBERT_study_new")
    text_tokens = tokenizer(["This is a test text.", "This is another test text."]).to("cuda:0")
    output = model(torch.randn(1, 1024).to("cuda:0"), text_tokens)
    print("Output:", output)
    print(model.clip.logit_scale)