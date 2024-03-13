import torch
from torchesn.nn import ESN
from get_training_data import x_train, y_train
from device import device

# Assuming x_train is of shape [N, seq_len, feature_size]
# Assuming y_train is of shape [N, feature_size]

# Initialize model parameters
input_size = 3  # Adjust if different from actual input size
hidden_size = 1000
output_size = 3  # Adjust if different from actual output size
washout_rate = 0.2  # Assuming a 20% washout rate to ignore initial transient

# Create the Echo State Network (ESN) model
model = ESN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
model = model.to(device)

# Convert training data to tensors and send to device
x_train_tensor = x_train.to(device)
y_train_tensor = y_train.to(device)

# Reshape y_train to match ESN output (if needed)
# This code assumes that the last dimension of y_train needs to match the output size
if y_train.shape[1] != output_size:
    raise ValueError("y_train feature size does not match output_size of the model")

# Determine the length of the washout period
washout_length = int(washout_rate * x_train_tensor.shape[1])

# Train the ESN
# Note: We are not washing out here as we do not loop over batches
print('HERER')
washout_list = [washout_length] * x_train_tensor.size(0)

model(x_train_tensor, washout_list, None, y_train_tensor)

# After presenting the training data once, call fit to train the readout
print('LÖKSJÖLFIJ')
model.fit()

# Evaluate the ESN
# For this example, we use the training data for simplicity
# To evaluate, you'd typically have separate validation or test data
print('ALÖIEJFÖ ')
output, _ = model(x_train_tensor, washout_length)

# Assuming a Mean Squared Error loss for evaluation
mse_loss = torch.nn.MSELoss()
print('LAKSEJF')
loss = mse_loss(output[washout_length:], y_train_tensor)

print("Training MSE Loss:", loss.item())