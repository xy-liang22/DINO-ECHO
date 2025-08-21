import json

with open('clip_study_keys.json', 'r') as f:
    clip_keys = json.load(f)
with open('dinov2_keys.json', 'r') as f:
    dinov2_keys = json.load(f)
clip_keys = set([k.replace('visual.vit.', '') for k in clip_keys if 'visual.vit.' in k])
clip_keys = set([k.replace('module.', '') for k in clip_keys])
clip_keys = set([k.replace('backbone.', '') for k in clip_keys])
dinov2_keys = set([k.replace('backbone.', '') for k in dinov2_keys])
# print('clip_keys:', clip_keys)
# print('dinov2_keys:', dinov2_keys)
print('clip_keys - dinov2_keys:', clip_keys - dinov2_keys)
print('dinov2_keys - clip_keys:', dinov2_keys - clip_keys)
with open('dinov2_keys_short.json', 'w') as f:
    json.dump(list(dinov2_keys), f)