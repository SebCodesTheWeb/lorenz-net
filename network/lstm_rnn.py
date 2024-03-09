import torch
import torch.nn as nn
from device import device as default_device

class LSTM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device=default_device):
        """
        input_size: input feature size, in this case 3 for the Lorenz system
        output_size: output feature size, in this case 3 for the Lorenz system
        hidden_size: number of hidden units in the LSTM
        """
        super(LSTM_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.output_activation = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, inputSeq):
        # inputSeq shape [batch_size, seq_len, feature_size]
        batch_size = inputSeq.size(0)
        state = self.init_state(batch_size)
        lstm_out, _ = self.lstm(inputSeq, state)

        # Select the last point in the sequence
        prediction = self.output_activation(lstm_out[:, -1, :])

        return prediction

    def init_state(self, batch_size):
        # Initializing the hidden and cell states for the LSTM based on the batch size
        state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
        )
        return state
