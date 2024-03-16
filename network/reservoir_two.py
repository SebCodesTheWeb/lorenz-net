from reservoirpy.datasets import lorenz
import matplotlib.pyplot as plt
from reservoirpy.observables import nrmse, rsquare
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
import json
from reservoirpy.hyper import research, plot_hyperopt_report
from constants import dt


timesteps = 2000
x0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]
X = lorenz(timesteps, x0=x0, method="RK45", h=dt)

fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("Lorenz")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid(False)

for i in range(timesteps-1):
    ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color=plt.cm.cividis(255*i//timesteps), lw=1.0)

plt.show()

train_len = 1000

X_train = X[:train_len]
y_train = X[1 : train_len + 1]

X_test = X[train_len : -1]
y_test = X[train_len + 1:]

dataset = ((X_train, y_train), (X_test, y_test))

def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):

    # This step may vary depending on what you put inside 'dataset'
    train_data, validation_data = dataset
    X_train, y_train = train_data
    X_val, y_val = validation_data

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []
    r2s = []
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=variable_seed)

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        # Train your model and test your model.
        predictions = model.fit(X_train, y_train) \
                           .run(X_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

hyperopt_config = {
    "exp": f"hyperopt-multiscroll", # the experimentation name
    "hp_max_evals": 200,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "N": ["choice", 500],             # the number of neurons is fixed to 300
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "lr": ["loguniform", 1e-3, 1],  # idem with the leaking rate, from 1e-3 to 1
        "iss": ["choice", 0.9],           # the input scaling is fixed
        "ridge": ["choice", 1e-7],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}

# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
# with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
#     json.dump(hyperopt_config, f)

# best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")

# fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr"), metric="r2")
# plt.show()

with open(f"hyperopt-multiscroll/results/0.0288120_hyperopt_results_1call.json", "r") as f:
    best_params = json.load(f)['current_params']

# Assuming that 'best_params' contains the best hyperparameters for the ESN
N = best_params['N']
sr = best_params['sr']
lr = best_params['lr']
iss = best_params['iss']
ridge_param = best_params['ridge']
seed = best_params['seed']

# Rebuild the model using the best hyperparameters
reservoir = Reservoir(N, sr=sr, lr=lr, input_scaling=iss, seed=seed)
readout = Ridge(ridge=ridge_param)
model = reservoir >> readout

# Train the model with the training data
model.fit(X_train, y_train)

# Generate predictions using the trained model
predictions = model.run(X_test)

# Now let's visualize the results in a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Generated Lorenz System by ESN")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid(False)

# Plot the generated Lorenz system
pred_len = len(predictions)
for i in range(pred_len - 1):
    ax.plot(predictions[i:i+2, 0], predictions[i:i+2, 1], predictions[i:i+2, 2],
            color=plt.cm.viridis(i / pred_len), lw=1.0)

plt.show()