from lorenz import RK4
import pandas as pd
import numpy as np
from constants import seed_nbr, dt

np.random.seed(seed_nbr)

offset_len = 2000
chunk_len = 200
total_data_points = 1e7
nbr_chunks = int(total_data_points // chunk_len)

dataset = []

initial_positions = np.random.rand(nbr_chunks, 3)

for i, pos in enumerate(initial_positions):
    print(i, len(initial_positions))
    # Offset each initial chunk to decorrelate the data
    for _ in range(offset_len):
        pos = RK4(pos, dt)
    
    # Generate the actual data chunk
    for j in range(chunk_len):
        elapsedTime = j * dt + offset_len * dt
        pos = RK4(pos, dt)
        x, y, z = pos
        dataset.append({
            't': elapsedTime,
            'x': x,
            'y': y,
            'z': z
        })

dataset = pd.DataFrame(dataset)

dataset.to_csv('lorentz-sequences_raw.csv', index=False)

# Normalize dataset
numerical_cols = ['x', 'y', 'z']
dataset[numerical_cols] = (
    dataset[numerical_cols] - dataset[numerical_cols].mean()
) / dataset[numerical_cols].std()

# Save to CSV
dataset.to_csv('lorentz-sequences.csv', index=False)