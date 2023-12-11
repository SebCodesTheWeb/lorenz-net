import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from get_training_data import x_test, y_test
from lstm_rnn import LSTM_RNN
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch

# print(f'Shape of x_test: {x_test.shape}')      # Expected to be (num_samples, sequence_length, input_dimension)
# print(f'Shape of y_test: {y_test.shape}')      # Expected to be (num_samples, output_dimension)

# Initialize your test data loader
hidden_size = 50
num_layers = 1
batch_size = 64

# Remove the unnecessary middle dimension from x_test and y_test
x_test = x_test.squeeze(1)  # Should now be [39999, 3]
y_test = y_test.squeeze(1)  # Should now be [39999, 3]

# Recreate the test DataLoader with the adjusted y_test tensor
test_data = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Load your model
input_size = x_test.shape[1]
output_size = y_test.shape[1]
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


# Plot the actual vs predicted in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot actual path
ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label='Actual Path', color='b')

# Plot predicted path
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predicted Path', color='r', linestyle='dashed')

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Display the plot
plt.title('3D Actual vs Predicted Paths')
plt.show()