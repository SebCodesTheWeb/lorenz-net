import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from get_training_data import x_test, y_test
from lstm_rnn import LSTM_RNN
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch

hidden_size = 50
num_layers = 1
batch_size=64

# No longer squeezing x_test and y_test
test_data = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

input_size = x_test.shape[2]  # Assuming x_test is of shape [num_samples, sequence_length, input_dimension]
output_size = y_test.shape[2]  # Assuming y_test is of shape [num_samples, 1, output_dimension]

model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm_rnn_lorenz.path'))

model.eval()  # Set model to evaluation mode
actual = []
predictions = []
with torch.no_grad():
    for seq, labels in test_dataloader:
        seq = seq.to(device)
        output = model(seq)
        # Reshape labels to remove the sequence_length dimension assuming it is 1
        labels_reshaped = labels.reshape(-1, output_size)
        actual.append(labels_reshaped.cpu().numpy())
        predictions.append(output.cpu().numpy())

# Flatten the lists of arrays into single numpy arrays
actual = np.concatenate(actual, axis=0)
predictions = np.concatenate(predictions, axis=0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot actual path
ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label='Actual Path', color='b')

# Plot predicted path
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predicted Path', color='r', linestyle='dashed')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.title('3D Actual vs Predicted Paths')
plt.show()