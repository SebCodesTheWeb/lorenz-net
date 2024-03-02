from get_transformer_training_data import x_train, y_train
from transformer import TransformerModel
from torch import nn
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch

# Hyperparams
hidden_dim = 500
nhead = 2
num_layers = 2
learning_rate = 0.0005
batch_size = 64
#Currently does not actually care
vocab_size = 3
# d_model has to be even due to how positional encoding is implemented, requiring pairs of sin and cosine positions
d_model = 128
dropout = 0.1

train_data = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

model = TransformerModel(ntoken=vocab_size, d_model=d_model, nhead=nhead, d_hid=hidden_dim, nlayers=num_layers,dropout=dropout).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    running_loss = 0.0

    for batch_nbr, (seq, label) in enumerate(dataloader):
        seq , label = seq.to(device), label.to(device)
        prediction = model(seq)
        # Prediction shape: torch.Size([batch_size=100, seq_len=500, feature_size=3]), Label shape: torch.Size([batch_size, feature_size])
        prediction = prediction[:, -1, :] # Selects the last point the in predicted sequence


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
    train(train_dataloader, model, loss_fn, optimizer)
    scheduler.step()
print("Done")
torch.save(model.state_dict(), 'transformer_lorenz.path')