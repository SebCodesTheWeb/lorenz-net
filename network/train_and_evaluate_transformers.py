from train_transformer import train_transformer
#from evaluate_networks import evaluate_model
from true_loss import evaluate_model

model = train_transformer()
evaluate_model(model)
