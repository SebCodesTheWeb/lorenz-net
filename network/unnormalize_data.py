import pandas as pd

dataset = pd.read_csv("lorenz-sequences_raw.csv")

mean_x, std_x = dataset["x"].mean(), dataset["x"].std()
mean_y, std_y = dataset["y"].mean(), dataset["y"].std()
mean_z, std_z = dataset["z"].mean(), dataset["z"].std()


def get_unnormalized_prediction(predictions: list):
    unnormalized_predictions = [
        [(x * std_x) + mean_x, (y * std_y) + mean_y, (z * std_z) + mean_z]
        for x, y, z in predictions
    ]

    return unnormalized_predictions
