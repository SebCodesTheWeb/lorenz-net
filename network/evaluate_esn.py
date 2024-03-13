import numpy as np
import torch
from constants import inp_seq_len, dt, seed_nbr
from lorenz import RK4
from train_network import train_rnn_lstm
from lstm_rnn import LSTM_RNN
from transformer import TransformerModel
from mse import calculate_aggregate_mse
from unnormalize_data import get_unnormalized_prediction
from device import device as default_device

np.random.seed(3)

def validate_long_term(nbrIter, init_input_seq, model, device):
    predicted_path = [*init_input_seq]
    for _ in range(nbrIter):
        input_seq = np.array(predicted_path[-inp_seq_len:])
        input_seq_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            next_pos, _ = model(input_seq_tensor)

        next_pos = next_pos.cpu().numpy().reshape(-1)
        predicted_path.append(next_pos)

    true_path = [*init_input_seq]
    for _ in range(nbrIter):
        true_path.append(RK4(np.array(true_path[-1]), dt).tolist())

    unnormalized_path = true_path[:inp_seq_len] + get_unnormalized_prediction(
        predicted_path[inp_seq_len:]
    )

    total = 0
    for i in range(3):
        list_a = [x[i] for x in true_path[inp_seq_len:]]
        list_b = [x[i] for x in unnormalized_path[inp_seq_len:]]
        mse_loss_ag = calculate_aggregate_mse(list_a, list_b)
        print(mse_loss_ag)
        total += mse_loss_ag
    total /= 3

    return  total

init_seq = [np.random.rand(3)]
for _ in range(inp_seq_len - 1):
    next_pos = RK4(np.array(init_seq[-1]), dt)
    init_seq.append(next_pos.tolist())

def evaluate_esn(model, device=default_device):
    loss = validate_long_term(
        1000,
        init_seq,
        model,
        device,
    )
    print(f"Validation loss: {loss}")


# import torch
# from get_training_data import x_val, y_val
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from device import device as default_device

# loss_fn = nn.MSELoss()
# val_data = TensorDataset(x_val, y_val)

# def evaluate_esn(model, device=default_device, batch_size=100):
#     dataloader = DataLoader(val_data, batch_size=batch_size)
#     model.eval()
#     running_loss = 0.0

#     with torch.no_grad():
#         for seq, label in dataloader:
#             seq, label = seq.to(device), label.to(device)
#             #Tuple unpacking(very important!)
#             prediction, _ = model(seq)
#             loss = loss_fn(prediction, label)
#             running_loss += loss.item()

#     running_loss /= len(dataloader)
#     print(f"Validation loss: {running_loss}")
#     return running_loss

