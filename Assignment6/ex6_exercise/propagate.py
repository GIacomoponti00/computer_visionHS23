import numpy as np


# This function should propagate the particles given the system prediction model (matrix A) and the
# system model noise represented by params.model, params.sigma position and params.sigma velocity.
# Use the parameter frame height and frame width to make sure that the center of the particle lies
# inside the frame.

def propagate (particles, frame_height, frame_width, params):
    if params["model"] == 0:
      #no motion model

      #dynamic matrix A
        A = np.identity(2)
        #noise vector w
        w = np.random.multivariate_normal([0,0,0,0], np.identity(4), params["num_particles"])
        #propagate particles
        #A has size mismatch with particles

        particles = np.matmul(A, particles.T).T + w
        #check if particles are out of bounds
        particles[:, 0] = np.clip(particles[:, 0], 0, frame_width)
        particles[:, 1] = np.clip(particles[:, 1], 0, frame_height)
        particles[:, 2] = np.clip(particles[:, 2], 0, frame_width)
        particles[:, 3] = np.clip(particles[:, 3], 0, frame_height)

        return particles
    elif params["model"] == 1:
        #constant velocity motion model

        #dynamic matrix A
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        #noise vector w
        w = np.random.multivariate_normal([0,0,0,0], np.identity(4), params["num_particles"])
        #propagate particles
        particles = np.matmul(A, particles.T).T + w
        #check if particles are out of bounds
        particles[:, 0] = np.clip(particles[:, 0], 0, frame_width)
        particles[:, 1] = np.clip(particles[:, 1], 0, frame_height)
        particles[:, 2] = np.clip(particles[:, 2], 0, frame_width)
        particles[:, 3] = np.clip(particles[:, 3], 0, frame_height)

    return particles