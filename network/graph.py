from lorenz import RK4
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
init_pos = np.random.rand(3)
path = [init_pos]
for _ in range(10000):
    new_pos = RK4(path[-1], dt)
    path.append(new_pos)

fig = plt.figure(figsize=(15, 10))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("Generated attractor")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.grid(False)

x_coords = [p[0] for p in path]
y_coords = [p[1] for p in path]
z_coords = [p[2] for p in path]
N = len(path)

for i in range(N-1):
    ax.plot(x_coords[i:i+2], y_coords[i:i+2], z_coords[i:i+2], color=plt.cm.magma(255*i//N), lw=1.0)
    
plt.show()