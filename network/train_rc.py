from get_training_data  import x_train, y_train
from rc_esn import EchoStateNetwork
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch

# Hyperparams
learning_rate = 0.0005
batch_size = 64
# input and output size must be three(x, y ,z)
input_size, output_size = 3, 3
reservoir_hidden_size = 1000
spectral_radius = 0.9
sparsity = 0.01
ridge_param = 1e-6

train_data = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

model = EchoStateNetwork(
    input_size=input_size,
    reservoir_size=reservoir_hidden_size,
    output_size=output_size,
    spectral_radius=spectral_radius,
    sparsity=sparsity,
).to(device)

def train_esn_with_ridge_regression(dataloader, model, ridge_param):
    # Concatenate all the reservoir states and desired_outputs
    reservoir_states = []
    desired_outputs = []
    
    # We won't need to track gradients for this process
    with torch.no_grad():
        for inputs, outputs in dataloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            
            # Use the last reservoir state for each sequence in the batch
            model_output, states = model(inputs)
            last_states = states[:, -1, :]  # Get the state from the last time step of each sequence
            
            reservoir_states.append(last_states)
            desired_outputs.append(outputs)
            
    # Concatenate all the reservoir_states and desired_outputs along the 0th dimension
    reservoir_states = torch.cat(reservoir_states, dim=0)
    desired_outputs = torch.cat(desired_outputs, dim=0)
    
    # Calculate the Ridge Regression solution
    identity_mat= torch.eye(model.reservoir_size, device=device)
    output_weights = torch.linalg.solve(
        reservoir_states.t() @ reservoir_states + ridge_param * identity_mat,
        reservoir_states.t() @ desired_outputs
    )
    
    # Set the model's output_weights to the computed Ridge Regression solution
    model.output_weights.data = output_weights.t()

train_esn_with_ridge_regression(train_dataloader, model, ridge_param)

print("Done")
torch.save(model.state_dict(), "rc_esn_lorenz.path")
