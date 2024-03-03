from train_network import train_rnn_lstm
from train_transformer import train_transformer
from evaluate_networks import evaluate_model

model = train_rnn_lstm()
evaluate_model(model)
