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

