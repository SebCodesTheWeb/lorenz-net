from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import torch
from device import device
from lstm_rnn import LSTM_RNN
from constants import dt
from lorenz import RK4
from transformer import TransformerModel
from rc_esn import EchoStateNetwork
from constants import inp_seq_len
from unnormalize_data import get_unnormalized_prediction

app = Flask(__name__)
CORS(app)

hidden_size = 128
num_layers = 1
input_size, output_size = 3, 3
d_model = 128
n_head = 2

rnn_model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)
rnn_model.load_state_dict(torch.load("lstm_rnn_lorenz.path"))

transformer_model = TransformerModel(
    d_model=128, nhead=2, d_hid=500, nlayers=2, dropout=0.1
).to(device)
transformer_model.load_state_dict(torch.load("transformer_lorenz.path"))


rc_model = EchoStateNetwork(
    input_size=3,
    reservoir_size=1000,
    output_size=3,
    spectral_radius=0.9,
    sparsity=0.01,
).to(device)

rc_model.load_state_dict(torch.load("rc_esn_lorenz.path", map_location="cpu"))


@app.route("/predict", methods=["GET"])
def predict_path():
    print('ran')
    t = float(request.args.get("t"))
    init_pos = request.args.get("init_pos").split(",")
    init_pos = [float(x) for x in init_pos]
    path = [init_pos]

    for _ in range(inp_seq_len):
        next_pos = RK4(np.array(path[-1]), dt)
        path.append(next_pos.tolist())


    current_pos_tensor = (
        #Using np.array(path) seems to speed things up
        torch.tensor(np.array(path), dtype=torch.float32).to(device).unsqueeze(0)
    )

    num_steps = int(t / 0.05)

    with torch.no_grad():
        for i in range(num_steps):
            next_pos = rnn_model(current_pos_tensor).cpu().numpy()[0].tolist()
            path.append(next_pos)
            print(i, num_steps)

            current_pos_tensor = (
                torch.tensor(np.array(path[1:] + [next_pos]), dtype=torch.float32).to(device).unsqueeze(0)
            )

    return jsonify(get_unnormalized_prediction(path))


@app.route("/predict_w_rc_esn", methods=["GET"])
def rc_predict_path():
    print("ran rc")
    t = float(request.args.get("t"))
    init_pos = request.args.get("init_pos").split(",")
    init_pos = [float(x) for x in init_pos]
    current_pos_tensor = (
        torch.tensor([init_pos], dtype=torch.float32).to(device).unsqueeze(0)
    )

    num_steps = int(t / dt)
    path = [init_pos]

    with torch.no_grad():
        for _ in range(num_steps):
            next_pos = rnn_model(current_pos_tensor).cpu().numpy()[0].tolist()
            path.append(next_pos)

            current_pos_tensor = (
                torch.tensor([next_pos], dtype=torch.float32).to(device).unsqueeze(0)
            )

    return jsonify(path)


@app.route("/predict_w_transformer", methods=["GET"])
def transformer_predict_path():
    print("ran")
    t = float(request.args.get("t"))
    init_pos = request.args.get("init_pos").split(",")
    init_pos = [float(x) for x in init_pos]
    current_pos_tensor = torch.tensor([init_pos], dtype=torch.float32).to(device)

    num_steps = int(t / dt)
    path = [init_pos]

    with torch.no_grad():
        for _ in range(num_steps):
            next_pos = transformer_model(current_pos_tensor).cpu().numpy()[0].tolist()
            # next_pos = [
            #     item for sublist in next_pos for item in sublist
            # ]  
            path.append(next_pos)

            current_pos_tensor = torch.tensor([next_pos], dtype=torch.float32).to(
                device
            )

    print(path)
    return jsonify(path)


@app.route("/rk4_predict", methods=["GET"])
def rk4_predict_path():
    t = float(request.args.get("t"))
    init_pos = request.args.get("init_pos").split(",")
    init_pos = [float(x) for x in init_pos]
    num_steps = int(t / dt)
    path = [init_pos]

    for _ in range(num_steps):
        next_pos = RK4(np.array(path[-1]), dt)
        path.append(next_pos.tolist())

    return jsonify(path)


if __name__ == "__main__":
    app.run(debug=True)
