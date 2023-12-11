from get_training_data import x_train, y_train
from lstm_rnn import LSTM_RNN
from torch import nn
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch

#Hyperparams
hidden_size = 50  
num_layers = 1  
learning_rate = 0.0005
batch_size = 100

train_data = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

input_size = x_train.shape[2]  
output_size = y_train.shape[2] 
model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    running_loss = 0.0

    for batch_nbr, (seq, label) in enumerate(dataloader):
        seq, label = seq.to(device), label.to(device)
        label = label.squeeze(1)
        prediction = model(seq)
        loss = loss_fn(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_nbr % 100 == 0:
            current = batch_nbr * len(seq)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches
    print(f"Average loss for epoch: {running_loss:>7f}")

epochs = 5
for t in range(epochs):
    print(f"epoch {t + 1} \n--------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done")
torch.save(model.state_dict(), 'lstm_rnn_lorenz.path')
