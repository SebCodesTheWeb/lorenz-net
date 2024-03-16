import numpy as np

def normalize_vectors(vectors):
    mean_x = np.mean([vec[0] for vec in vectors])
    mean_y = np.mean([vec[1] for vec in vectors])
    mean_z = np.mean([vec[2] for vec in vectors])
    
    std_x = np.std([vec[0] for vec in vectors])
    std_y = np.std([vec[1] for vec in vectors])
    std_z = np.std([vec[2] for vec in vectors])
    
    normalized_vectors = [( (vec[0] - mean_x) / std_x,
                            (vec[1] - mean_y) / std_y,
                            (vec[2] - mean_z) / std_z) for vec in vectors]
    return normalized_vectors