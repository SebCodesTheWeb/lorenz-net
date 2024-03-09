from lorenz import RK4
import numpy as np

# Initial condition for the Lorenz system
initial_condition = np.array([1.0, 1.0, 1.0])

# Small perturbation
epsilon = 1e-8
perturbation = epsilon * np.random.normal(size=3)

# Initialize variables
dt = 0.005  # Time step size
max_iter = 200000  # Number of iterations to simulate
renorm_iter = 200  # Number of iterations after which to renormalize

# Main loop for Lyapunov exponent calculation
trajectory = initial_condition
perturbed_trajectory = initial_condition + perturbation
sum_log_d = 0

for i in range(max_iter):
    # Integrate both trajectories
    trajectory = RK4(trajectory, dt)
    perturbed_trajectory = RK4(perturbed_trajectory, dt)

    # Calculate the exponential growth of the error
    if i % renorm_iter == 0:
        d = perturbed_trajectory - trajectory
        norm_d = np.linalg.norm(d)
        sum_log_d += np.log(norm_d / epsilon)
        perturbed_trajectory = trajectory + (epsilon * d / norm_d)

# Calculate the largest Lyapunov exponent
largest_lyapunov_exp = sum_log_d / (max_iter * dt)

# Calculate the Lyapunov time
if largest_lyapunov_exp > 0:
    lyapunov_time = 1 / largest_lyapunov_exp
    print(f"Largest Lyapunov Exponent: {largest_lyapunov_exp}")
    print(f"Lyapunov Time: {lyapunov_time}")
    optimal_nbr_steps = (lyapunov_time / dt) / dt
    print(f"One Lyapunov time corresponds to {optimal_nbr_steps} time units of dt.")
else:
    print("The system may not be chaotic or the exponent may have been miscalculated.")