# graphs.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from get_training_data import x_test, y_test
from lstm_rnn import LSTM_RNN
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch

# Initialize your test data loader
hidden_size = 50
num_layers = 1
batch_size = 64

test_data = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Load your model
input_size = x_test.shape[2]
output_size = y_test.shape[2]
model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm_rnn_lorenz.path'))

# Running the model to get predictions
model.eval()  # Set model to evaluation mode
actual = []
predictions = []
with torch.no_grad():
    for seq, labels in test_dataloader:
        seq = seq.to(device)
        output = model(seq)
        actual.append(labels.cpu().numpy())
        predictions.append(output.cpu().numpy())

# Flatten the lists of arrays into single numpy arrays
actual = np.concatenate(actual, axis=0)
predictions = np.concatenate(predictions, axis=0)

# You can now use 'actual' and 'predictions' with plotting code from previous messages...
actual = y_test.cpu().numpy()
predictions = []
model.eval()  # Make sure model is in eval mode for inference
with torch.no_grad():
    for seq, _ in test_dataloader:
        seq = seq.to(device)
        prediction = model(seq).cpu().numpy()
        predictions.append(prediction)
predictions = np.vstack(predictions)

# Plot predictions vs actual for x, y, z over time
time_steps = np.arange(actual.shape[0])
for i, component in enumerate(['x', 'y', 'z']):
    plt.figure()
    plt.plot(time_steps, actual[:, i], label='Actual ' + component)
    plt.plot(time_steps, predictions[:, i], label='Predicted ' + component, linestyle='dashed')
    plt.xlabel('Time Step')
    plt.ylabel(component)
    plt.legend()
    plt.title(f'Actual vs Predicted {component} values over Time')
    plt.show()