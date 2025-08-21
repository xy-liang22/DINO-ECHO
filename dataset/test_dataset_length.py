import numpy as np
import pandas as pd
import os

# path = '/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/1.3.46.670589.52.2.540275.20181024.5120918.4471.17539/D2I3JO0G.npy'

# csv_path = '/home/patxiao/ECHO/label_dataset_v2/AVA.csv'
csv_path = '/mnt/hanoverdev/scratch/hanwen/xyliang/ECHO_dataset_csv/label_v4/LHF.csv'
df = pd.read_csv(csv_path)
paths = list(df['path'])
n = len(paths)
num = 100
indices = np.random.choice(n, num, replace=False)

total = 0
minn = float('inf')
maxx = 0
large_num = 0
l = []
total_num_videos_in_study = 0
for k, i in enumerate(indices):
    videos_dir = '/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/' + paths[i]
    study_len = 0
    total_num_videos_in_study += len(os.listdir(videos_dir))    
    for file in os.listdir(videos_dir):
        path = os.path.join(videos_dir, file)
        img = np.load(path)
        study_len += img.shape[1]
    total += study_len
    minn = min(minn, study_len)
    maxx = max(maxx, study_len)
    l.append(img.shape[1])
    if img.shape[1] > 500:
        large_num += 1
    if (k + 1) % 10 == 0:
        print(k + 1)

print(total / num, minn, maxx, large_num, total_num_videos_in_study / num)
print(l)