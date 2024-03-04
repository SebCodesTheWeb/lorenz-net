from train_rc import train_rc_esn
from evaluate_networks import evaluate_model

model = train_rc_esn()
evaluate_model(model)
