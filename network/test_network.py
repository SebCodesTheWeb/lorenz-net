from get_training_data import x_test, y_test
from lstm_rnn import LSTM_RNN
from torch import nn
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch

hidden_size = 50  
num_layers = 1  
batch_size=64

test_data = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

input_size = x_test.shape[2]  
output_size = y_test.shape[2] 
model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm_rnn_lorenz.path'))
loss_fn = nn.MSELoss()

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for seq, label in dataloader:
            seq, label = seq.to(device), label.to(device)
            prediction = model(seq)
            label = label.view_as(prediction)
            test_loss += loss_fn(prediction, label).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"epoch {t + 1} \n--------------")
    test(test_dataloader, model, loss_fn)
print("Done")