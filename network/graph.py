from lorenz import RK4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

dt = 0.01
init_pos = np.random.rand(3)
path = [init_pos]
for _ in range(10000):
    new_pos = RK4(path[-1], dt)
    path.append(new_pos)

# Increase the figsize for a larger figure and set the dpi for higher resolution
fig = plt.figure(figsize=(20, 15), dpi=200, facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')
ax.set_xlabel("$x$", color='black')
ax.set_ylabel("$y$", color='black')
ax.set_zlabel("$z$", color='black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.tick_params(axis='z', colors='black')
ax.grid(False)

# Making the panes transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Removing the axes' spines
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

x_coords = [p[0] for p in path]
y_coords = [p[1] for p in path]
z_coords = [p[2] for p in path]
N = len(path)
norm = Normalize(vmin=0, vmax=N)

# Color mapping for points
colors = plt.cm.inferno(norm(range(N)))

# Adjust the line width variation here, making the lines thinner
for i in range(N-1):
    lw = 0.06 + 0.4 * (i / (N-1))  # Reduced maximum line width for a thinner plot
    ax.plot(x_coords[i:i+2], y_coords[i:i+2], z_coords[i:i+2], color=colors[i], lw=lw)

# Hide the axis
ax.set_axis_off()

plt.show()