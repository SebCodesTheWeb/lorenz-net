from constants import seed_nbr, inp_seq_len, dt
from train_network import train_rnn_lstm
from device import device
from lstm_rnn import LSTM_RNN
import torch
import numpy as np
from lorenz import RK4
import matplotlib.pyplot as plt
from unnormalize_data import get_unnormalized_prediction
from transformer import TransformerModel

def mse_3d(point_a, point_b):
    return np.sum((np.array(point_a) - np.array(point_b))**2) / 3.0

def calculate_aggregate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must contain the same number of elements")
    
    mse_values = [mse_3d(point_a, point_b) for point_a, point_b in zip(list1, list2)]
    return np.mean(mse_values)

np.random.seed(seed_nbr + 5)

nbrTimeSteps = 5000
nbrIterations = 1

rnn_model = LSTM_RNN(input_size=3, output_size=3, hidden_size=512, num_layers=1).to(
    device
)
rnn_model.load_state_dict(torch.load("lstm_rnn_lorenz.path"))
rnn_model.eval()

# transformers_model = TransformerModel(
#     d_model=128, nhead=2, d_hid=500, nlayers=2, dropout=0
# ).to(device)
# transformers_model.load_state_dict(torch.load("transformer_lorenz.path"))
# transformers_model.eval()


init_positions = np.random.rand(nbrIterations, 3)
for i in range(nbrIterations):
    init_pos = init_positions[i]
    path = [init_pos]

    # Generate start input sequence
    for _ in range(inp_seq_len - 1):
        next_pos = RK4(np.array(path[-1]), dt)
        path.append(next_pos.tolist())

    # Run model for nbrTimeSteps
    model_path = [*path]
    for _ in range(nbrTimeSteps):
        # print(f"{model_path[-inp_seq_len:][-1]} \n")
        current_pos_tensor = (
            torch.tensor(np.array(model_path[-inp_seq_len:]), dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
        )
        # next_pos = rnn_model(current_pos_tensor).cpu().detach().numpy()[0].tolist()
        next_pos = (
            rnn_model(current_pos_tensor).cpu().detach().numpy()[0].tolist()
        )
        model_path.append(next_pos)

    # Continue with RK4 for nbrTimeSteps
    for _ in range(nbrTimeSteps):
        next_pos = RK4(np.array(path[-1]), dt)
        path.append(next_pos.tolist())

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Three subplots for x, y, z

    unnormalized_path = model_path[:inp_seq_len] + get_unnormalized_prediction(
        model_path[inp_seq_len:]
    )
    for i in range(3):
        coordModel = [point[i] for point in unnormalized_path]
        coordGroundZero = [point[i] for point in path]
        aggregate_mse = calculate_aggregate_mse(coordModel, coordGroundZero)
        print(f"The aggregate MSE for the entire lists is: {aggregate_mse:.4f}")

        x_axis = range(1, len(model_path) + 1)

        axs[i].plot(x_axis, coordModel, "bo-", label="List 1 (Blue)")
        axs[i].plot(x_axis, coordGroundZero, "ro-", label="List 2 (Red)")

        axs[i].set_xlabel("Time step")
        axs[i].set_ylabel(f"Coordinate {chr(120+i)}")

        axs[i].legend()

    plt.tight_layout()
    plt.show()
