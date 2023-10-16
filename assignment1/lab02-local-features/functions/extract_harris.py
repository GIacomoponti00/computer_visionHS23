import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

import cv2 as cv #manually added import 

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.06, thresh = 1e-4):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction

    # declare sharr kernel
    kernel_y = np.array([[-3, -10, -3], 
                       [0, 0, 0], 
                       [3, 10, 3]])
    kernel_x = np.array([[-3, 0, 3], 
                       [-10, 0, 10], 
                       [-3, 0, 3]])

    # compute the image gradients
    Ix = signal.convolve2d(img, kernel_x, boundary="symm", mode='same') # cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    Iy = signal.convolve2d(img, kernel_y, boundary="symm", mode='same') # cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    
    # 2. Blur the computed gradients

    #cv.GaussianBlur(Iy, Iy_blur, [3, 3], sigma, borderType=cv.BORDER_REPLICATE)
    #cv.GaussianBlur(Ix, Ix_blur, [3, 3], sigma, borderType=cv.BORDER_REPLICATE)
    Ixx = ndimage.gaussian_filter(np.square(Ix), sigma, mode='nearest')
    Ixy = ndimage.gaussian_filter(Ix*Iy, sigma, mode='nearest')
    Iyy = ndimage.gaussian_filter(np.square(Iy), sigma, mode='nearest')

    # 3. Compute elements of the local auto-correlation matrix "M"

    # 4. Compute Harris response function C
    C = (Ixx*Iyy - np.square(Ixy)) - (k*np.square(Ixx + Iyy))

    # 5. Detection with threshold and non-maximum suppression
    corners = np.stack(np.where(C > thresh*C.max()), axis=1)

    max_suppr = ndimage.maximum_filter(C, footprint=np.ones((3, 3), dtype=bool))
    for candidate in corners:
        if C[candidate[0], candidate[1]] != max_suppr[candidate[0], candidate[1]]:
            corners = np.delete(corners, np.where((corners == candidate).all(axis=1))[0], axis=0)

    corners[:, [1, 0]]= corners[:, [0, 1]]
    return corners, C

