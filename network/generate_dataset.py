from lorenz import RK4
import pandas as pd
import numpy as np
from constants import seed_nbr, dt, chunk_len

np.random.seed(seed_nbr)

total_data_points = 1e6
nbr_chunks = int(total_data_points // chunk_len)

pos = [17.67715816276679, 12.931379185960404, 43.91404334248268]

dataset = []
dataset.append({
    't': 0,
    'x': pos[0],
    'y': pos[1],
    'z': pos[2],
})

for i, _ in enumerate(nbr_chunks):
    for j in range(chunk_len):
        elapsedTime = j * dt + dt
        pos = RK4(pos, dt)
        x, y, z = pos
        dataset.append({
            't': elapsedTime,
            'x': x,
            'y': y,
            'z': z
        })

dataset = pd.DataFrame(dataset)

dataset.to_csv('lorenz-sequences_raw.csv', index=False)

#Z-score normalization, 
numerical_cols = ['x', 'y', 'z']
dataset[numerical_cols] = (
    dataset[numerical_cols] - dataset[numerical_cols].mean()

) / dataset[numerical_cols].std()

dataset.to_csv('lorentz-sequences.csv', index=False)