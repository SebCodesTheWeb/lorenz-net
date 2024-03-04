import optuna
from train_transformer import train_transformer
from train_network import train_rnn_lstm
from train_rc import train_rc_esn
from evaluate_networks import evaluate_model

model_type = "Transformer"


def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    epochs = 5

    if model_type == "Transformer":
        model_hyperparams = {
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", [256, 512, 768, 1024]
            ),
            "nhead": trial.suggest_int("nhead", 1, 8),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0, 0.5),
            "epochs": epochs,
        }

        model = train_transformer(model_hyperparams)
        val_loss = evaluate_model(model)

    elif model_type == "RNN_LSTM":
        model_hyperparams = {
            "hidden_size": trial.suggest_categorical(
                "hidden_dim_lstm", [32, 64, 128, 256, 512]
            ),
            "num_layers": trial.suggest_int("num_layers_lstm", 1, 3),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
        }
        model = train_rnn_lstm(model_hyperparams)
        val_loss = evaluate_model(model)

    elif model_type == "ESN":
        model_hyperparams = {
            "batch_size": batch_size,
            "input_size": 3,
            "output_size": 3,
            "reservoir_hidden_size": trial.suggest_categorical(
                "reservoir_hidden_size", [500, 1000, 1500]
            ),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
            "sparsity": trial.suggest_float("sparsity", 0, 0.5),
            "ridge_param": trial.suggest_float("ridge_param", 1e-8, 1e-4),
        }

        model = train_rc_esn(model_hyperparams)
        val_loss = evaluate_model(model)

    return val_loss


# Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(
    objective, n_trials=100, n_jobs=4
)  # This is assuming your setup allows parallelization across 4 GPUs
