from train_network import train_rnn_lstm
# from evaluate_networks import evaluate_model
from true_loss import evaluate_model

model = train_rnn_lstm()
evaluate_model(model)
