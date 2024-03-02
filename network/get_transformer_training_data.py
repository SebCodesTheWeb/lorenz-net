import pandas as pd
import torch

seq_len = 10

dataset = pd.read_csv('lorentz-sequences.csv')
data_tensor = torch.tensor(dataset[['x', 'y', 'z']].values, dtype=torch.float32)

x_train = []
y_train = []

for i in range(len(data_tensor) - seq_len):
    x_train.append(data_tensor[i:i + seq_len])
    y_train.append(data_tensor[i + seq_len])

x_train = torch.stack(x_train)  # dim: (N, seq_len, data_point_size=3)
y_train = torch.stack(y_train)  # dim: (N, data_point_size=3)