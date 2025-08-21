import os
import sys
import timm
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision import transforms

from open_clip import create_model_and_transforms


def make_echoclip(model_name):
    ### build the echoclip model ###
    print(f"Start loading {model_name} model")
    model, _, _ = create_model_and_transforms(model_name, device="cuda")
    print(f"Finish loading {model_name} model")
    
    return model
    

class EchoClipAsClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int=1, 
                 hidden_dim: int=512,
                 **kwargs):
        '''
        Implement the EchoClip model for ECHO classification task.

        Arguments:
        ----------
        num_classes (int): number of classes
        hidden_dim (int): hidden dimension
        '''
        
        super(EchoClipAsClassifier, self).__init__()
        self.image_encoder = make_echoclip("hf-hub:mkaichristensen/echo-clip-r")
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.size = self.image_encoder.visual.image_size
        mean = torch.tensor(self.image_encoder.visual.image_mean or (0.48145466, 0.4578275, 0.40821073))
        std = torch.tensor(self.image_encoder.visual.image_std or (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Normalize(mean=mean, std=std)
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}
    
    def make_3ch(self, x):
        '''Make 3 channel input'''
        if x.shape[1] == 1:
            return torch.cat([x, x, x], dim=1)
        else:
            return x
    
    def forward(self, x):
        assert len(x.shape) == 5
        B, C, T, H, W = x.shape

        # reshape input [B, C, T, H, W] -> [B * T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.make_3ch(x)
        
        x = F.interpolate(x, size=self.size, mode="bicubic", align_corners=False)
        
        # normalize the input
        if x.max() > 1:
            x = x / 255.0
        
        x = self.transform(x)
        
        # run through the backbone
        embedding_output = self.image_encoder.encode_image(x) # [B * T, hidden_dim]
        embedding_output = embedding_output.view(B, T, -1) # [B, T, hidden_dim]
        
        # mean pooling
        h = torch.mean(embedding_output, dim=1) # [B, hidden_dim]
        
        # classifier
        logits = self.classifier(h)
        
        return logits


def echoclip_cnn_classifier(**kwargs):
    model = EchoClipAsClassifier(hidden_dim=512, **kwargs)
    return model


if __name__ == '__main__':
    model = echoclip_cnn_classifier(num_classes=2).cuda()
    
    # x = torch.randn(1, 1, 64, 256, 256).cuda()
    # with torch.no_grad():
    #     with torch.cuda.amp.autocast():
    #         out = model(x)
    # print(out.shape)