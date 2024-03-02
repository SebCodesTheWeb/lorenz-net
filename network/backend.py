from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import torch
from device import device
from lstm_rnn import LSTM_RNN
from constants import dt
from lorenz import RK4, eulers_method
from transformer import TransformerModel

app = Flask(__name__)
CORS(app)

hidden_size = 50
num_layers = 1
input_size, output_size = 3, 3
vocab_size = 3
d_model = 128
n_head= 2

rnn_model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)
rnn_model.load_state_dict(torch.load('lstm_rnn_lorenz.path'))

transformer_model = TransformerModel(ntoken=3, d_model=128, nhead=2, d_hid=500, nlayers=2, dropout=0.1).to(device)
transformer_model.load_state_dict(torch.load('transformer_lorenz.path'))

@app.route("/predict", methods=['GET'])
def predict_path():
    t = float(request.args.get('t'))
    init_pos = request.args.get('init_pos').split(',')
    init_pos = [float(x) for x in init_pos]
    current_pos_tensor = torch.tensor([init_pos], dtype=torch.float32).to(device).unsqueeze(0)

    num_steps = int(t / dt)
    path = [init_pos]

    with torch.no_grad():
        for _ in range(num_steps):
            next_pos = rnn_model(current_pos_tensor).cpu().numpy()[0].tolist()
            path.append(next_pos)

            current_pos_tensor = torch.tensor([next_pos], dtype=torch.float32).to(device).unsqueeze(0)

    return jsonify(path)


@app.route("/predict_w_transformer", methods=['GET'])
def transformer_predict_path():
    print('ran')
    t = float(request.args.get('t'))
    init_pos = request.args.get('init_pos').split(',')
    init_pos = [float(x) for x in init_pos]
    current_pos_tensor = torch.tensor([init_pos], dtype=torch.float32).to(device)

    num_steps = int(t / dt)
    path = [init_pos]

    with torch.no_grad():
        for _ in range(num_steps):
            next_pos = transformer_model(current_pos_tensor).cpu().numpy()[0].tolist()
            next_pos = [item for sublist in next_pos for item in sublist]  # Flatten the list
            path.append(next_pos)

            current_pos_tensor = torch.tensor([next_pos], dtype=torch.float32).to(device)

    print(path)
    return jsonify(path)

@app.route("/rk4_predict", methods=['GET'])
def rk4_predict_path():
    t = float(request.args.get('t'))
    init_pos = request.args.get('init_pos').split(',')
    init_pos = [float(x) for x in init_pos]
    num_steps = int(t / dt)
    path = [init_pos]

    for _ in range(num_steps):
        next_pos = RK4(np.array(path[-1]), dt)
        path.append(next_pos.tolist())

    return jsonify(path)


if __name__ == "__main__":
    app.run(debug=True)
