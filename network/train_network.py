from get_training_data import x_train, y_train
from lstm_rnn import LSTM_RNN
from torch import nn
from device import device as default_device
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import ExponentialLR
import optuna

def train_rnn_lstm(
    hidden_size=100,
    num_layers=1,
    learning_rate=0.0005,
    batch_size=8,
    epochs=10,
    trial = None,
    gpu_id=0,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else default_device)

    train_data = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    input_size = x_train.shape[2]
    output_size = y_train.shape[1]
    model = LSTM_RNN(input_size, hidden_size, output_size, num_layers, device).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.7)

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
                print(f"loss: {loss}  [{current:>5d}/{size:>5d}]")

        running_loss /= num_batches
        print(f"Average loss for epoch: {running_loss}")
        return running_loss

    for t in range(epochs):
        print(f"epoch {t + 1} \n--------------")
        loss = train(train_dataloader, model, loss_fn, optimizer)
        trial.report(loss, t)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        scheduler.step()
    print("Done")
    torch.save(model.state_dict(), "lstm_rnn_lorenz.path")

    return model
