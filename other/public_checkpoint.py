import torch
import json
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# torch.save(model.state_dict(), '/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_public.pt')

model_path = "/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_public.pt"
output_path = "dinov2_vitl14.json"
model = torch.load(model_path)
keys = list(model.keys())
with open(output_path, 'w') as f:
    json.dump(keys, f)
print(f"Model keys saved to {output_path}")