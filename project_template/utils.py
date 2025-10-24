import numpy as np
import yaml
config = yaml.safe_load(open('config.yml','r'))

def get_data(n_samples):
    rand = np.random.default_rng(42)
    x = rand.random(size=(n_samples, config['N_FEATURES']))
    y = rand.random(size=n_samples)
    return x,y