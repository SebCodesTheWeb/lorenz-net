import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lorenz import RK4


# Set initial conditions and time step
initial_conditions = np.array([1.0, 1.0, 1.0])
dt = 0.01

# Define number of iterations
steps = 10000

# Create arrays to store x, y and z positions
xs, ys, zs = np.empty(steps), np.empty(steps), np.empty(steps)
xs[0], ys[0], zs[0] = initial_conditions

# Run the RK4 method
for i in range(1, steps):
    xs[i], ys[i], zs[i] = RK4(np.array([xs[i-1], ys[i-1], zs[i-1]]), dt)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Lorenz attractor
ax.plot(xs, ys, zs, lw=0.5)

# Set labels
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor using RK4")

# Show the plot
plt.show()