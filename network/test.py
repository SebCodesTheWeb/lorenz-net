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

np.random.seed(seed_nbr + 5)

nbrTimeSteps = 5000
nbrIterations = 1

rnn_model = LSTM_RNN(input_size=3, output_size=3, hidden_size=128, num_layers=1).to(
    device
)
rnn_model.load_state_dict(torch.load("lstm_rnn_lorenz.path", map_location="mps"))
rnn_model.eval()

# transformers_model = TransformerModel(
#     d_model=128,
#     nhead=2,
#     d_hid=500,
#     nlayers=2,
#     dropout=0
# ).to(device)
# transformers_model.load_state_dict(torch.load("transformer_lorenz.path", map_location="mps"))
# transformers_model.eval()



init_positions = np.random.rand(nbrIterations, 3)
for i in range(nbrIterations):
    init_pos = init_positions[i]
    path = [init_pos]

    # Generate start input sequence
    for _ in range(inp_seq_len - 1 ):
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
        next_pos = rnn_model(current_pos_tensor).cpu().detach().numpy()[0].tolist()
        model_path.append(next_pos)

    # Continue with RK4 for nbrTimeSteps
    for _ in range(nbrTimeSteps):
        next_pos = RK4(np.array(path[-1]), dt)
        path.append(next_pos.tolist())

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Three subplots for x, y, z

    for i in range(3):
        coordModel = [
            point[i]
            # for point in model_path[: inp_seq_len - 1]
            # + get_unnormalized_prediction(model_path[inp_seq_len - 1 :])
            for point in get_unnormalized_prediction(model_path)
        ]
        coordGroundZero = [point[i] for point in path]

        x_axis = range(1, len(model_path) + 1)

        axs[i].plot(x_axis, coordModel, "bo-", label="List 1 (Blue)")
        axs[i].plot(x_axis, coordGroundZero, "ro-", label="List 2 (Red)")

        axs[i].set_xlabel("Time step")
        axs[i].set_ylabel(f"Coordinate {chr(120+i)}")

        axs[i].legend()

    plt.tight_layout()
    plt.show()
