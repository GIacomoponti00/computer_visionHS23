import numpy as np

def estimate (particles, particles_w):
    mean_state = np.average(particles, weights=particles_w, axis=0)
    return mean_state