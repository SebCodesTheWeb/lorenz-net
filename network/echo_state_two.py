from get_training_data  import x_train, y_train
# from rc_esn import EchoStateNetwork
from device import device as default_device
from torch.utils.data import DataLoader, TensorDataset
import torch
# import echotorch as etnn
import echotorch.nn.reservoir as etnn
from torch.autograd import Variable

def train_rc_esn(
    batch_size=64,
    input_size=3,
    output_size=3,
    reservoir_hidden_size=1000,
    spectral_radius=0.9,
    sparsity=0.01,
    ridge_param=1e-6,
    input_scaling=100,
    input_weights_scaling=0.01,
    device=default_device
):
    x_train_device = x_train.to(device)
    y_train_device = y_train.to(device)

    train_data = TensorDataset(x_train_device, y_train_device)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    esn_model = etnn.LiESN(
        input_dim=input_size,
        n_hidden=reservoir_hidden_size,
        output_dim=output_size,
        spectral_radius=spectral_radius,
        learning_algo='inv',
        leaky_rate=1,
        sparsity=sparsity,
    ).to(device)

    def train(dataloader, model):
        with torch.no_grad():
            for inputs, outputs in dataloader:
                inputs, outputs = inputs.to(device), outputs.to(device)

                inputs, outputs = Variable(inputs), Variable(outputs)

                model(inputs, outputs)


    train(train_dataloader, esn_model)
    esn_model.finalize()
    print("Done")

    return esn_model