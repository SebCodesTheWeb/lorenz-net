import pandas as pd
import torch
import numpy as np
from constants import inp_seq_len, test_ratio, seed_nbr

dataset = pd.read_csv('lorentz-sequences.csv')
data_tensor = torch.tensor(dataset[['x', 'y', 'z']].values, dtype=torch.float32)

def create_seq(input):
    seq = []
    for i in range(len(input) - inp_seq_len):
        chunk = input[i:i + inp_seq_len]
        label = input[i + inp_seq_len:i + inp_seq_len+1]
        seq.append((chunk, label))
    return seq

data_seq = create_seq(data_tensor)


def manual_train_test_split(data, test_ratio, seed):
    np.random.seed(seed)
    num_examples = len(data)
    split_idx = int(num_examples * test_ratio)
    shuffled_indices = np.random.permutation(num_examples)
    
    test_indices, train_indices = shuffled_indices[:split_idx], shuffled_indices[split_idx:]
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, test_data

inout_seq_train, inout_seq_test = manual_train_test_split(data_seq, test_ratio, seed_nbr)

x_train = torch.stack([item[0] for item in inout_seq_train])
y_train = torch.stack([item[1] for item in inout_seq_train])
x_test = torch.stack([item[0] for item in inout_seq_test])
y_test = torch.stack([item[1] for item in inout_seq_test])
