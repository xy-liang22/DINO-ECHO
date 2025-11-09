import pandas as pd
from tqdm import tqdm
import torch

embedding_path = "/data/ECHO/dinov2_study_original1_embeddings_multi_videos_mean.pt"
embeddings = torch.load(embedding_path)
split = 'test'

output_path = '/data/ECHO/llava_data_label/task_predict.csv'
data = {'split': list(), 'path': list()}

for study in tqdm(embeddings):
    data['split'].append(split)
    data['path'].append(study)

data = pd.DataFrame(data)
data.to_csv(output_path, index=False)
print(f"Save to {output_path}")

