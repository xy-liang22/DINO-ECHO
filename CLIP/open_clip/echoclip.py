import os
import sys
import timm
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision import transforms

from .factory import create_model_and_transforms


class EchoCLIPVisual(nn.Module):
    def __init__(self,
                 echoclip_path: str,
                 **kwargs):
        '''
        Implement the DINOv2 model with attention mechanism for ECHO classification task.

        Arguments:
        ----------
        config_path (str): path to the DINOv2 config file
        num_classes (int): number of classes
        pretrained (str): path to the pretrained model
        hidden_dim (int): hidden dimension
        '''
        
        super(EchoCLIPVisual, self).__init__()
        model, _, preprocess_val = create_model_and_transforms(
            echoclip_path,
            precision="bf16",
            device="cuda"
        )
        print(f"Using EchoCLIP model from: {echoclip_path}")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "vit.cls_token",
            "vit.pos_embed",
            "vit.pos_embed_spatial",
            "vit.pos_embed_temporal",
            "vit.pos_embed_class",
        }
    
    def make_3ch(self, x):
        '''Make 3 channel input'''
        if x.shape[1] == 1:
            return torch.cat([x, x, x], dim=1)
        else:
            return x
    
    def forward(self, x):
        x, masks = x
        assert len(x.shape) == 5
        B, C, F, H, W = x.shape

        # reshape input [B, C, F, H, W] -> [B * F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        x = self.make_3ch(x)
        
        # normalize the input
        if x.max() > 1:
            x = x / 255.0
        
        x = self.transform(x)
        
        # run through the backbone
        embedding_output = self.vit(x) # [B * F, hidden_dim]
        embedding_output = embedding_output.view(B, F, -1) # [B, F, hidden_dim]
        embedding_output = embedding_output * masks.unsqueeze(-1)
        logits = embedding_output.sum(dim=1) / masks.sum(dim=1, keepdim=True)  # [B, hidden_dim]
        
        return logits


if __name__ == '__main__':
    pretrained = "/mnt/hanoverdev/scratch/hanwen/exp/echofound/pretrain_dinov2/20250205_vitl16_lbsz64_gbsz512_500ep_noKoleo/eval/training_624999/teacher_checkpoint.pth"
    model = DINOv2Mean(config_path='open_clip/dinov2_modules/configs/train/vitl16_lbsz48_short.yaml', pretrained=pretrained)
    x = torch.randn(4, 3, 64, 256, 256).cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = model(x)
    print(out.shape)
