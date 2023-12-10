from lorenz import RK4  
import pandas as pd
import numpy as np

np.random.seed(0)
nbr_chunks = 100
chunk_len = 2000
dt = 0.005

dataset = []

for chunk_idx in range(nbr_chunks):
    init_pos = np.random.rand(3)
    pos = init_pos

    #offset each chunk by 2000 timsteps
    if chunk_idx > 0:
        for _ in range(2000):
            pos = RK4(pos, dt)
    
    for i in range(chunk_len):
        elapsedTime = i * dt
        pos = RK4(pos, dt)
        x, y, z = pos
        dataset.append({
            't': elapsedTime,
            'x': x,
            'y': y,
            'z': z
        })
    
dataset = pd.DataFrame(dataset)

#Normalize dataset
numerical_cols = ['x', 'y', 'z']
dataset[numerical_cols] = (
    dataset[numerical_cols] - dataset[numerical_cols].mean()
    ) / dataset[numerical_cols].std()

dataset.to_csv('lorentz-sequences.csv', index=False)
