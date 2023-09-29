import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

import cv2 as cv #manually added import 

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
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
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    # declare the kernel for computing the image gradients
    kernel = np.array([[-1, 0, 1]])
    # compute the image gradients
    Ix = signal.convolve2d(img, kernel, mode='same')
    Iy = signal.convolve2d(img, kernel.T, mode='same')
    #print("finished step 1")
    # raise NotImplementedError
    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    Iy_blur = cv.GaussianBlur(Iy, ksize=(0, 0), sigmaX=sigma, borderType=cv.BORDER_REPLICATE)
    Ix_blur = cv.GaussianBlur(Ix, ksize=(0, 0), sigmaX=sigma, borderType=cv.BORDER_REPLICATE)
    #print("finished step 2")
    # raise NotImplementedError

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    Ix2 = np.matmul(Ix_blur, Ix_blur) # Ix_blur * Ix_blur
    #print(np.shape(Ix2))
    Iy2 = np.matmul(Iy_blur, Iy_blur) # Iy_blur * Iy_blur 
    IxIy = np.matmul(Ix_blur, Iy_blur)
    #print(np.shape(IxIy))
    M = np.block([[Ix2, IxIy], 
                  [IxIy, Ix2]])
    
    #print("finished step 3, size of M: ")
    #print(np.shape(M))
    
    # raise NotImplementedError

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    #C = np.linalg.det(M) - k*pow(np.trace(M), 2)
    C = np.matmul(Ix2, Iy2) - IxIy - k*np.linalg.matrix_power((Ix2 + Iy2), 2)
    # print(C1.size) 
    # print("finished step 4")
    # raise NotImplementedError

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    
    #max_suppressed = ndimage.maximum_filter(C, size=3)
    #print(np.shape(C))
    corners = np.stack(np.where(C >= thresh))

    print("finished step 5")

    #raise NotImplementedError
    return corners, C

