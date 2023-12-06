import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


def distance(x, X):
    distances = np.sqrt(np.sum((x - X)**2, axis=1))
    return distances


def gaussian(dist, bandwidth):
    return np.exp(-0.5 * (dist / bandwidth)**2) / (bandwidth * np.sqrt(2 * np.pi))    


def update_point(weight, X):
    return np.sum(weight.reshape(-1, 1) * X, axis=0) / np.sum(weight)

def meanshift_step(X, bandwidth=3):
    for i in range(X.shape[0]):
        dist = distance(X[i], X)
        weight = gaussian(dist, bandwidth)
        X[i] = update_point(weight, X)
    return X

def meanshift(X):
    for _ in range(20):
        print('Iteration {}'.format(_))
        X = meanshift_step(X)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('../mean-shift/eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
