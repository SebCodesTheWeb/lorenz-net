import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json


def mse_3d(point_a, point_b):
    return np.sum((np.array(point_a) - np.array(point_b)) ** 2) / 3.0


def calculate_aggregate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must contain the same number of elements")

    mse_values = [mse_3d(point_a, point_b) for point_a, point_b in zip(list1, list2)]
    return np.mean(mse_values)

# Read the data back from the file
with open('latest_path.json', 'r') as file:
    data = json.load(file)

model_path = data['model']
ground_zero_path = data['rk4']

fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # Three subplots for x, y, z
for i in range(3):
    coordModel = [point[i] for point in model_path]
    coordGroundZero = [point[i] for point in ground_zero_path]
    aggregate_mse = calculate_aggregate_mse(coordModel, coordGroundZero)
    print(f"The aggregate MSE for the entire lists is: {aggregate_mse:.4f}")

    x_axis = range(1, len(model_path) + 1)

    axs[i].plot(x_axis, coordModel, "bo-", label="List 1 (Blue)")
    axs[i].plot(x_axis, coordGroundZero, "ro-", label="List 2 (Red)")

    axs[i].set_xlabel("Time step")
    axs[i].set_ylabel(f"Coordinate {chr(120+i)}")

    axs[i].legend()

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))

# Create two side-by-side 3D subplots
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

# Set the axes limits and labels
ax1.set_xlim([-30, 30])
ax1.set_ylim([-30, 30])
ax1.set_zlim([0, 30])
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2.set_xlim([-30, 30])
ax2.set_ylim([-30, 30])
ax2.set_zlim([0, 30])
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

# Initialize two lines for animation
(line1,) = ax1.plot([], [], [], lw=2, label="RK4")
(line2,) = ax2.plot([], [], [], lw=2, label="RNN Model")

# Initialization function for FuncAnimation
def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    return line1, line2

# Update function for FuncAnimation
def update(num, pathA, pathB, line1, line2):
    # Update the data of both lines with the new frame
    line1.set_data(pathA[:num, 0], pathA[:num, 1])
    line1.set_3d_properties(pathA[:num, 2])
    line2.set_data(pathB[:num, 0], pathB[:num, 1])
    line2.set_3d_properties(pathB[:num, 2])
    return line1, line2

# Number of frames (adjust as needed)
num_frames = len(ground_zero_path)

# Create the animation
ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    fargs=(np.array(ground_zero_path), np.array(model_path), line1, line2),
    init_func=init,
    blit=True,
    interval=50,
)

# Show the animation
plt.legend()
plt.show()

# Calculate the MSE for each pair of points
mse_values_over_time = [mse_3d(point_a, point_b) for point_a, point_b in zip(ground_zero_path, model_path)]

# Create a new figure for the MSE plot
fig_mse, ax_mse = plt.subplots(figsize=(10, 6))

# Plot the MSE over time
ax_mse.plot(mse_values_over_time, "g-", label="MSE over Time")

# Label the axes and add a legend
ax_mse.set_xlabel("Time step")
ax_mse.set_ylabel("MSE")
ax_mse.set_title("MSE for each point in 3D over time")
ax_mse.legend()

# Display the plot
plt.show()