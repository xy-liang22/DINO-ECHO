import torch
from torch import nn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dinov2_as_classifier import make_dinov2
from torchvision import transforms


# checkpoint_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer.pt"
# dinov2_only_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer_DINO_only.pt"
# config_path = "/home/xuhw/xyliang/research-projects/ECHO/models/dinov2_modules/configs/train/vitl16_lbsz48_short.yaml"

checkpoint_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer_original1.pt"
dinov2_only_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer_original1_DINO_only.pt"
config_path = "/home/xuhw/xyliang/research-projects/ECHO/models/dinov2_modules/configs/train/vitl16_lbsz96_500ep_resume_from_general.yaml"
class DINOv2(nn.Module):
    def __init__(self,
                 config_path: str,
                 image_size: int=256,
                 pretrained: str=None,
                 embed_dim: int=768,
                 is_study: bool=False,
                 use_transformer: bool=False,
                 width: int=1024,
                 proj: bool=True,
                 n_heads: int=8,
                 transformer_layers: int=4,
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
        
        super(DINOv2, self).__init__()
        assert width == 1024, "Only width == 1024 is supported for now."
        self.image_size = image_size
        self.vit, _ = make_dinov2(config_path, ckpt_path=pretrained)
        self.transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.is_study = is_study
        self.use_transformer = use_transformer
        if use_transformer:
            self.cls_token = nn.Parameter(torch.randn(1, 1, width))
            self.positional_embedding = nn.Parameter(
                torch.randn(1, 128, width)  # Max sequence length 1024
            )
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=n_heads,
                dim_feedforward=width * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layers,
                num_layers=transformer_layers,
                norm=nn.LayerNorm(width)
            )
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Temporal pooling for video data
        self.proj = None
        if width != embed_dim or proj:
            hidden_size = (width + embed_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(width, hidden_size, bias=False),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, embed_dim, bias=False),
            )
        print(f"Using DINOv2Mean model with config: {config_path}, pretrained: {pretrained}")
        print(f"❗❗❗Using transformer: {use_transformer}, is_study: {is_study}❗❗❗")

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
        if self.is_study:
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
        
        if self.is_study:
            # apply masks if provided
            embedding_output = embedding_output * masks.unsqueeze(-1)
            if self.use_transformer:
                cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
                embedding_output = torch.cat((cls_tokens, embedding_output), dim=1)  # [B, F + 1, hidden_dim]
                embedding_output = embedding_output + self.positional_embedding[:, :embedding_output.size(1), :]  # [B, F + 1, hidden_dim]
                masks = torch.cat((torch.ones(B, 1, device=embedding_output.device), masks), dim=1)  # [B, F + 1]
                encoded_embeddings = self.transformer(embedding_output, src_key_padding_mask=~masks.bool())
                logits = encoded_embeddings[:, 0, :]  # [B, hidden_dim]
                temporal_rep = self.temporal_pool(encoded_embeddings[:, 1:, :].permute(0, 2, 1)).squeeze(2)  # [B, hidden_dim, 1]
                logits = logits + temporal_rep  # [B, hidden_dim]
            else:
                logits = embedding_output.sum(dim=1) / masks.sum(dim=1, keepdim=True)  # [B, hidden_dim]
        else:
            # mean pooling
            logits = torch.mean(embedding_output, dim=1) # [B, hidden_dim]
        
        if self.proj is not None:
            logits = self.proj(logits)
        
        return logits

model = DINOv2(
    config_path=config_path,
    image_size=224,
    pretrained=dinov2_only_path,
    embed_dim=1024,
    use_transformer=True,
    is_study=True
)

msg = model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
print("Loaded DINOv2 model with message:", msg)