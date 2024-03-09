from lorenz import RK4
import pandas as pd
import numpy as np
from constants import seed_nbr, dt, chunk_len

np.random.seed(seed_nbr)

#Keep low, just seems to make things worse
offset_len = 0
total_data_points = 5e7
nbr_chunks = int(total_data_points // chunk_len)
len_before_reset = 1e6

dataset = []

initial_positions = np.random.rand(nbr_chunks, 3) 
pos = initial_positions[0]

for i, _ in enumerate(initial_positions):
    # print(i, len(initial_positions))
    pos = initial_positions[i] if i * chunk_len % len_before_reset == 0 else pos
    if(i*chunk_len % len_before_reset == 0):
        print(len(dataset))
    # Offset each initial chunk to decorrelate the data
    for _ in range(offset_len):
        pos = RK4(pos, dt)
   
    # Generate the actual data chunk
    for j in range(chunk_len):
        elapsedTime = j * dt + i * offset_len * dt
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

numerical_cols = ['x', 'y', 'z']
dataset[numerical_cols] = (
    dataset[numerical_cols] - dataset[numerical_cols].mean()

) / dataset[numerical_cols].std()

# Save to CSV
dataset.to_csv('lorentz-sequences.csv', index=False)