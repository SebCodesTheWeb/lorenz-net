import torch
import torch.nn as nn
from device import device

class LSTM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_activation = nn.Linear(hidden_size, output_size)
    
    def forward(self, inputSeq):
        batch_size = inputSeq.size(0)
        state = self.init_state(batch_size)
        lstm_out, _ = self.lstm(inputSeq, state)

        prediction = self.output_activation(lstm_out[:, -1, :])

        return prediction

    def init_state(self, batch_size):
        state = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        return state