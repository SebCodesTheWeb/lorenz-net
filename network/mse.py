
import numpy as np 

def mse_3d(point_a, point_b):
    return np.sum((np.array(point_a) - np.array(point_b)) ** 2) / 3.0


def calculate_aggregate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must contain the same number of elements")

    mse_values = [mse_3d(point_a, point_b) for point_a, point_b in zip(list1, list2)]
    return np.mean(mse_values)
