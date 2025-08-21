import os
import sys
import timm
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision import transforms

class LinearClassifier(nn.Module):
    def __init__(self,
                 num_classes: int=1,
                 hidden_dim: int=1024,
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
        
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        print(f"LinearClassifier: num_classes={num_classes}, hidden_dim={hidden_dim}")

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()
    
    def get_num_layers(self):
        return 1
    
    def forward(self, x):
        return self.classifier(x)
        

def linear_classifier(**kwargs):
    model = LinearClassifier(**kwargs)
    return model


if __name__ == '__main__':
    model = linear_classifier(num_classes=12).cuda()
    x = torch.randn(256, 1024).cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = model(x)
    print(out.shape)
