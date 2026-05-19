import numpy as np
for i in range(1, 11):
    a = np.load(f'data/train/subject{i:02d}.npz')
    print(f'subject{i:02d}: x={a["x"].shape} y={a["y"].shape} classes={np.bincount(a["y"])}')
t = np.load('data/test.npz')
print(f'test     : x={t["x"].shape} id={t["id"].shape if "id" in t else None}')