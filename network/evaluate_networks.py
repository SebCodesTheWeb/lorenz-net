import torch
from get_training_data import x_val, y_val
from device import device
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

loss_fn = nn.MSELoss()
val_data = TensorDataset(x_val, y_val)

def evaluate_model(model, batch_size=100):
    dataloader = DataLoader(val_data, batch_size=batch_size)
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for seq, label in dataloader:
            seq, label = seq.to(device), label.to(device)
            prediction = model(seq)
            loss = loss_fn(prediction, label)
            running_loss += loss.item()

    running_loss /= len(dataloader)
    print(f"Validation loss: {running_loss:>7f}")

