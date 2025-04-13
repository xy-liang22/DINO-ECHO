import numpy as np
import pandas as pd

# path = '/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/1.3.46.670589.52.2.540275.20181024.5120918.4471.17539/D2I3JO0G.npy'

csv_path = '/home/patxiao/ECHO/label_dataset_v2/AVA.csv'
df = pd.read_csv(csv_path)
paths = df['path'].to_numpy()
n = len(paths)
num = 1000
indices = np.random.choice(n, num, replace=False)

total = 0
minn = float('inf')
maxx = 0
large_num = 0
l = []
for k, i in enumerate(indices):
    path = '/mnt/hanoverdev/data/patxiao/ECHO_numpy/20250126/' + paths[i]
    img = np.load(path)
    total += img.shape[1]
    minn = min(minn, img.shape[1])
    maxx = max(maxx, img.shape[1])
    l.append(img.shape[1])
    if img.shape[1] > 128:
        large_num += 1
    if (k + 1) % 100 == 0:
        print(k + 1)

print(total / num, minn, maxx, large_num)
print(l)