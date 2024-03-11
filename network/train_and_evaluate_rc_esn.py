from train_rc import train_rc_esn
from evaluate_esn import evaluate_esn

model = train_rc_esn()
evaluate_esn(model)
