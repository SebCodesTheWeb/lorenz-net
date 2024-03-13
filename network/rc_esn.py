import torch
from torch import Tensor
import torch.nn as nn
from constants import seed_nbr

# torch.manual_seed(seed_nbr)

class EchoStateNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        reservoir_size,
        output_size,
        spectral_radius=0.95,
        sparsity=0.02,
        input_scaling=100,
        input_weights_scaling=0.01,
        noise_level=0.01,
    ):
        """
        input_size: feature size, in this case 3 for the Lorenz system
        reservoir_size: number of neurons in the reservoir
        spectral_radius: largest absolute eigenvalue of the reservoir matrix
        sparsity: percentage of the reservoir connected nodes
        """
        super(EchoStateNetwork, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.input_weights_scaling = input_weights_scaling
        self.noise_level = noise_level

        # Input weights and reservoir weights are not trained
        self.input_weights = nn.Parameter(
            torch.randn(reservoir_size, input_size) * input_scaling, requires_grad=False
        )
        self.reservoir_weights = nn.Parameter(
            self.generate_reservoir(reservoir_size, spectral_radius, sparsity),
            requires_grad=False,
        )

        # Output weights will be trained
        self.output_weights = nn.Parameter(
            torch.randn(output_size, reservoir_size), requires_grad=True
        )

    def generate_reservoir(self, size, spectral_radius, sparsity):
        random_reservoir = (torch.rand(size, size) - 0.5) * self.input_weights_scaling
        mask = torch.rand(size, size) > sparsity
        # Mask is applied with 1-sparsity percentage of the random weights set to 0
        random_reservoir[mask] = 0

        # Scale the reservoir to have the desired spectral radius
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(random_reservoir)
            actual_spectral_radius = max(abs(eigenvalues))
            random_reservoir *= spectral_radius / actual_spectral_radius
        return random_reservoir

    def forward(self, input: Tensor):
        # input shape: [batch_size, seq_len, feature_size]
        states = []
        # state shape: [batch_size, reservoir_size]
        state = torch.zeros(input.size(0), self.reservoir_size).to(input.device)
        noise = torch.randn_like(state) * self.noise_level
        for t in range(input.size(1)):
            state = torch.tanh(
                self.input_weights @ input[:, t].t()
                + self.reservoir_weights @ state.t()
                + noise.t()
            ).t()
            states.append(state.unsqueeze(1))

        states_tensor = torch.cat(states, dim=1)

        outputs = self.output_weights @ states_tensor[:, -1, :].t()

        return outputs.t(), states_tensor
