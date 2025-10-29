import numpy as np

def get_data(n_samples, n_features):
    rand = np.random.default_rng(42)
    x = rand.random(size=(n_samples, n_features))
    y = rand.random(size=n_samples)
    return x,y