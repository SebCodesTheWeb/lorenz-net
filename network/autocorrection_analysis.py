import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('lorentz-sequences.csv')

# Define the features for which to perform autocorrelation analysis
features = ['x', 'y', 'z']

# Create a figure to hold the subplots
plt.figure(figsize=(18, 6))

# Loop through the features and create a subplot for each feature's autocorrelation plot
for i, feature in enumerate(features, 1):
    plt.subplot(1, len(features), i)  # Arguments are (rows, columns, subplot index)
    pd.plotting.autocorrelation_plot(dataset[feature][:10000])
    plt.title(f'Autocorrelation of {feature}')

# Show the plot
plt.tight_layout()
plt.show()
