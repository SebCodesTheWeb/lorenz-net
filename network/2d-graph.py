# graphs.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_training_data import x_test, y_test
from lstm_rnn import LSTM_RNN
from device import device
from torch.utils.data import DataLoader, TensorDataset

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

# ... (Your code for DataLoader and model initialization)

actual = []
predictions = []
with torch.no_grad():
    for seq, labels in test_dataloader:
        seq = seq.to(device)
        output = model(seq).cpu()
        actual.extend(labels.cpu())
        predictions.extend(output)

# Now make sure to convert lists to numpy arrays
actual = np.vstack(actual)
predictions = np.vstack(predictions)

# You can now proceed to plot 'x' components
# Extract the 'x' component
actual_x = actual[:, 0]
predictions_x = predictions[:, 0]

# Plot the x-component graph
plt.figure(figsize=(14, 4))  # Increase the figure size to stretch it in the x dimension
plt.plot(actual_x, label='Actual x', color='blue')
plt.plot(predictions_x, label='Predicted x', linestyle='dashed', color='red')
plt.xlabel('Time Step')
plt.ylabel('X component')
plt.legend()
plt.title('Comparison of Actual vs Predicted for X component')
plt.show()