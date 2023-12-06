import numpy as np
from chi2_cost import chi2_cost


# This function should resample the particles based on their weights (eq. 6.), and return these new
# particles along with their corresponding weights.

def resample (particles, particles_w):
    for particle in particles:

        #the weights are modeled by a normal probability distribution with 
        #variance sigma_resample and mean equal to chi2_cost of the target histogram and the current histogram
        particle[4] = np.random.normal(chi2_cost(particle[4], particles_w), 0.1)

    #normalize weights
    particles_w = particles[:, 4]
    particles_w = particles_w / np.sum(particles_w)

    #resample particles
    particles = np.random.choice(particles, size=particles.shape[0], replace=True, p=particles_w)

    return particles, particles_w