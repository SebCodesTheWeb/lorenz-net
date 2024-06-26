import optuna
import torch
from train_transformer import train_transformer
from train_network import train_rnn_lstm
from true_loss import evaluate_model
import csv
from device import device as default_device

model_type = "Transformer"

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128, 256])
    # epochs = trial.suggest_int('epochs', 5, 10)
    gpu_id = trial.number % 4
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else default_device
    )

    if model_type == "Transformer":
        model_hyperparams = {
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", [256, 512, 768, 1024]
            ),
            "nhead": trial.suggest_int("nhead", 1, 2),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0, 0.4),
            "epochs": 5,
            "trial": trial,
            "device": device,
        }

        model = train_transformer(**model_hyperparams)
        val_loss = evaluate_model(model, device)
        return val_loss

    elif model_type == "RNN_LSTM":
        model_hyperparams = {
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256, 512]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": 5,
            "gamma": trial.suggest_float("gamma", 0.7, 1),
            "trial": trial,
            "device": device,
        }

        model = train_rnn_lstm(**model_hyperparams)
        val_loss = evaluate_model(model, device)
        return val_loss


# Optuna study
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(
    direction="minimize", pruner=pruner, storage="sqlite:///lorenz_study.db"
)
study.optimize(
    objective, n_trials=100, n_jobs=4, show_progress_bar=True
)  # n_jobs is number of parallel jobs(one per gpu available)

# Print the best trial's hyperparameters
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params_filename = "best_trial_params.csv"
with open(best_params_filename, mode="w", newline="") as csvfile:
    fieldnames = ["parameter", "value"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in trial.params.items():
        writer.writerow({"parameter": key, "value": value})

print(f"Best trial parameters saved to {best_params_filename}")

print("Saving best params by re-training...")
best_model = train_rnn_lstm(**trial.params)
