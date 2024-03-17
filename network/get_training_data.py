import pandas as pd
import torch
import numpy as np
from constants import (
    inp_seq_len,
    val_ratio,
    seed_nbr,
)

dataset = pd.read_csv("lorentz-sequences.csv")
data_tensor = torch.tensor(dataset[["x", "y", "z"]].values, dtype=torch.float32)

test_set = data_tensor[-10000:].numpy().tolist()
data_tensor = data_tensor[:-10000]


def create_seq(data):
    seq = []
    for i in range(0, len(data) - inp_seq_len, inp_seq_len):
        input = data[i : i + inp_seq_len]
        label = data[i + inp_seq_len]
        # Given input sequence, predict the next value(label)
        seq.append((input, label))
    return seq


data_seq = create_seq(data_tensor)


def manual_split(data, val_ratio, seed):
    np.random.seed(seed)
    num_examples = len(data)
    val_size = int(num_examples * val_ratio)

    shuffled_indices = np.random.permutation(num_examples)
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    return train_data, val_data


inout_seq_train, inout_seq_val = manual_split(data_seq, val_ratio, seed_nbr)

# item[0] is input sequence, item[1] is label
x_train = torch.stack([item[0] for item in inout_seq_train])
y_train = torch.stack([item[1] for item in inout_seq_train])
x_val = torch.stack([item[0] for item in inout_seq_val])
y_val = torch.stack([item[1] for item in inout_seq_val])
