import torch
import json

# path = '/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs/BiomedBERT_study_lr3e-6_frame80_bsz2_accum12_dist/checkpoints/epoch_24.pt'
# path = '/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs/BiomedBERT_study_lr3e-6_frame80_bsz2_accum12_proj_transformer_dist/checkpoints/epoch_25.pt'
path = '/mnt/hanoverdev/scratch/hanwen/xyliang/CLIP_logs/BiomedBERT_study_original1_proj_dist/checkpoints/epoch_20.pt'
# path = '/mnt/hanoverdev/scratch/hanwen/exp/echofound/pretrain_dinov2/20250205_vitl16_lbsz64_gbsz512_500ep_noKoleo/eval/training_624999/teacher_checkpoint.pth' # for dinov2
state_dict = torch.load(path)
state_dict = state_dict['state_dict']

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'module.' in k}
state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if 'visual.' in k}

keys_path = 'clip_study_original1_keys.json'
with open(keys_path, 'w') as f:
    keys = list(state_dict.keys())
    json.dump(keys, f)
    print('saved keys to', keys_path)

# save_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_transformer_original1.pt'
# torch.save(state_dict, save_path)
# print('saved to', save_path)

# state_dict = {k.replace('visual.vit.', ''): v for k, v in state_dict.items() if 'visual.vit.' in k}
# ckpt = dict(teacher=state_dict)

state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items() if 'vit.' in k}

keys_path = 'clip_study_original1_keys_dinov2.json'
with open(keys_path, 'w') as f:
    keys = list(state_dict.keys())
    json.dump(keys, f)
    print('saved keys to', keys_path)

ckpt = dict(teacher=state_dict)
save_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/DINOv2_study_original1.pt'
torch.save(ckpt, save_path)
print('saved to', save_path)
