import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram
from estimate import estimate

def observe(particles, frame , bbox_height, bbox_width, params_hist_bin, hist, params_sigma_observe):
     for particle in particles:
        center = np.array([particle[0] + 0.5 * particle[2], particle[1] + 0.5 * particle[3]])
        hist_x = color_histogram(center[0] - 0.5 * bbox_width, 
                                    center[1] - 0.5 * bbox_height, 
                                    center[0] + 0.5 * bbox_width,
                                    center[1] + 0.5 * bbox_height, 
                                    frame, params_hist_bin)
        particle[4] = chi2_cost(hist_x, hist) / params_sigma_observe
        particles_w = particles[:, 4]
        particles_w = particles_w / np.sum(particles_w)
        return particles_w
