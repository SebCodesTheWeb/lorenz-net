from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from device import device
from lstm_rnn import LSTM_RNN
from constants import dt

app = Flask(__name__)
CORS(app)

hidden_size = 50
num_layers = 1
input_size, output_size = 3, 3
model = LSTM_RNN(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm_rnn_lorenz.path'))

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
            next_pos = model(current_pos_tensor).cpu().numpy()[0].tolist()
            path.append(next_pos)

            current_pos_tensor = torch.tensor([next_pos], dtype=torch.float32).to(device).unsqueeze(0)

    return jsonify(path)


if __name__ == "__main__":
    app.run(debug=True)