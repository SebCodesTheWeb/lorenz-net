import reservoirpy as rpy
# from constants import seed_nbr, dt
from reservoirpy.nodes import Reservoir, Ridge, NVAR
import matplotlib.pyplot as plt
import numpy as np
from reservoirpy.datasets import lorenz

rpy.verbosity(999)
rpy.set_seed(0)

dt = 0.025
train_time = 10.
test_time = 120.
warm_time = 5.

train_steps = round(train_time / dt)
test_steps  = round(test_time  / dt)
warm_steps  = round(warm_time  / dt)

# x0 = np.random.rand(3)
#Seems like good starting position for training data
x0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]
n_timesteps = train_steps + test_steps + warm_steps

X = lorenz(n_timesteps, x0=x0, h=dt, method="RK45")

reservoir = Reservoir(1000,  lr=0.5, sr=0.9)
nvar = NVAR(delay=2, order=2, strides=1)
readout = Ridge(3, ridge=2.5e-6)
esn_model = nvar >> readout

X_train = X[:train_steps + warm_steps -1]
Y_train = X[1:train_steps + warm_steps] - X[:train_steps + warm_steps -1]

model = esn_model.fit(X_train, Y_train, warmup=warm_steps)


nvar.run(X[warm_steps+train_steps-2:warm_steps+train_steps])
u = X[warm_steps+train_steps]
res = np.zeros((test_steps, readout.output_dim))
for i in range(test_steps):
    u = u + model(u)
    res[i, :] = u

N = test_steps
Y = X[warm_steps+train_steps:]

fig = plt.figure(figsize=(15, 10))
ax  = fig.add_subplot(121, projection='3d')
ax.set_title("Generated attractor")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.grid(False)

for i in range(N-1):
    ax.plot(res[i:i+2, 0], res[i:i+2, 1], res[i:i+2, 2], color=plt.cm.magma(255*i//N), lw=1.0)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Real attractor")
ax2.grid(False)

for i in range(N-1):
    ax2.plot(Y[i:i+2, 0], Y[i:i+2, 1], Y[i:i+2, 2], color=plt.cm.magma(255*i//N), lw=1.0)

plt.show()


# fig = plt.figure(figsize=(10, 10))
# ax  = fig.add_subplot(111, projection='3d')
# ax.set_title("Lorenz attractor (1963)")
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
# ax.set_zlabel("$z$")
# ax.grid(False)

# for i in range(n_timesteps-1):
#     ax.plot(X[i:i+2, 0], X[i:i+2, 1], X[i:i+2, 2], color=plt.cm.magma(255*i//n_timesteps), lw=1.0)

# plt.show()


# reservoir = Reservoir(100,  lr=0.5, sr=0.9)

# # linspace creates linearly spaced number between 0 and 6*pi over 100 points, reshape shapes it into a 2d column
# X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
# readout = Ridge(ridge=1e-7)

# esn_model = reservoir >> readout

# X_train = X[:50]
# Y_train = X[1:51] #Shifted by 1

# esn_model.fit(X_train, Y_train, warmup=10)

# warmup_y = esn_model.run(X_train[:-10], reset=True)


# Y_pred = np.empty((100, 1))
# x = warmup_y[-1].reshape(1, -1)

# for i in range(100):
#     x = esn_model(x)
#     Y_pred[i] = x

# plt.figure(figsize=(10, 3))
# plt.title("100 timesteps of a sine wave.")
# plt.xlabel("$t$")
# plt.plot(Y_pred, label="Generated sin(t)")
# plt.legend()
# plt.show()

# # Y_pred = esn_model.run(X[50:])


# # train_states = reservoir.run(X_train, reset=True)

# # readout = readout.fit(train_states, Y_train, warmup=10)

# # test_states = reservoir.run(X[50:])
# # Y_pred = readout.run(test_states)

# # plt.figure(figsize=(10, 3))
# # plt.title("A sine wave and its future.")
# # plt.xlabel("$t$")
# # plt.plot(Y_pred, label="Predicted sin(t)", color="blue")
# # plt.plot(X[51:], label="Real sin(t+1)", color="red")
# # plt.legend()
# # plt.show()

# # s = reservoir(X[0].reshape(1, -1))

# # states = np.empty((len(X), reservoir.output_dim))

# # for i in range(len(X)):
# #     states[i] = reservoir(X[i].reshape(1, -1))
# # a_state_vector = np.random.uniform(-1, 1, size=(1, reservoir.output_dim))

# # states = reservoir.run(X, reset=True, from_state=a_state_vector)

# # plt.figure(figsize=(10, 3))
# # plt.title("Activation of 20 reservoir neurons.")
# # plt.ylabel("$reservoir(sin(t))$")
# # plt.xlabel("$t$")
# # plt.plot(states[:, :20])
# # plt.show()
# # s = reservoir.state()
# # print("New state vector shape: ", s.shape)


# # plt.figure(figsize=(10, 3))
# # plt.title("A sine wave.")
# # plt.ylabel("$sin(t)$")
# # plt.xlabel("$t$")
# # plt.plot(X)
# # plt.show()

# # print('running fine')