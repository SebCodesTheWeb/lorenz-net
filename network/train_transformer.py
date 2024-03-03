from get_training_data import x_train, y_train
from transformer import TransformerModel
from torch import nn
from device import device
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch


def train_transformer(
    hidden_dim=500,
    nhead=2,
    num_layers=2,
    learning_rate=0.0005,
    batch_size=64,
    d_model=128,
    dropout=0,
    epochs=5,
):
    assert (
        d_model % 2 == 0
    ), "d_model must be an even number! This is due to how positional encoding is implemented."

    train_data = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    model = TransformerModel(
        d_model=d_model,
        nhead=nhead,
        d_hid=hidden_dim,
        nlayers=num_layers,
        dropout=dropout,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.train()
        running_loss = 0.0

        for batch_nbr, (seq, label) in enumerate(dataloader):
            seq, label = seq.to(device), label.to(device)
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

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        scheduler.step()
    print("Done")
    torch.save(model.state_dict(), "transformer_lorenz.path")

    return model
