import optuna

model_type = "Transformer"


def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Model-specific training and evaluation
    if model_type == "Transformer":
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
        nhead = trial.suggest_int("nhead", 1, 8)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0, 0.5)

        model_hyperparams = {
            "hidden_dim": hidden_dim,
            "nhead": nhead,
            "num_layers": num_layers,
            "d_model": d_model,
            "dropout": dropout,
        }

        model = load_and_train_transformer(model_hyperparams)
        val_loss = evaluate_transformer(model)
    elif model_type == "RNN_LSTM":
        hidden_dim = trial.suggest_categorical("hidden_dim_lstm", [128, 256, 512])
        num_layers = trial.suggest_int("num_layers_lstm", 1, 3)
        dropout = trial.suggest_float("dropout_lstm", 0, 0.5)

        model_hyperparams = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        model = load_and_train_rnn_lstm(model_hyperparams)
        val_loss = evaluate_rnn_lstm(model)
    elif model_type == "ESN":
        reservoir_size = trial.suggest_categorical("reservoir_size", [500, 1000, 1500])
        spectral_radius = trial.suggest_float("spectral_radius", 0.5, 1.5)
        sparsity = trial.suggest_float("sparsity", 0, 0.5)

        model_hyperparams = {
            "reservoir_size": reservoir_size,
            "spectral_radius": spectral_radius,
            "sparsity": sparsity,
        }

        model = load_and_train_esn(model_hyperparams)
        val_loss = evaluate_esn(model)

    return val_loss


# Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(
    objective, n_trials=100, n_jobs=4
)  # This is assuming your setup allows parallelization across 4 GPUs
