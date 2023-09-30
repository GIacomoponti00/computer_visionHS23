import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

import cv2 as cv #manually added import 

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.06, thresh = 1e-5):
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
    cv.imwrite("img.png", img)
    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    # declare the kernel for computing the image gradients, use getDerivKernels to get the row and column kernels
    # kernel = cv.getDerivKernels(1, 0, 3, normalize=True)
    # kernel = np.array([-1, 0, 1])
    #declare sobel kernel
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobel kernel
    # compute the image gradients
    Ix = signal.convolve2d(img, kernel, mode='same')
    cv.imwrite("Ix.png", Ix)
    kernel =  np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Iy = signal.convolve2d(img, kernel, mode='same')
    #   cv.imwrite("Iy.png", Iy)

    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    Iy2 = cv.GaussianBlur(Iy**2, ksize=(0, 0), sigmaX=sigma, borderType=cv.BORDER_REPLICATE)
    Ix2 = cv.GaussianBlur(Ix**2, ksize=(0, 0), sigmaX=sigma, borderType=cv.BORDER_REPLICATE)
    cv.imwrite("Ix_blur.png", Ix2)
    #cv.imwrite("Iy_blur.png", Iy_blur)



    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    #Ix2 = Ix_blur**2 # np.matmul(Ix_blur, Ix_blur) # np.power(Ix_blur, 2) 
    cv.imwrite("Ix2.png", Ix2)
    #Iy2 = Iy_blur**2 #np.matmul(Iy_blur, Iy_blur) # Iy_blur * Iy_blur 
    IxIy = cv.GaussianBlur(Ix*Iy, ksize=(0, 0), sigmaX=sigma, borderType=cv.BORDER_REPLICATE) # np.matmul(Ix_blur, Iy_blur)
    #print(np.shape(IxIy))
    M = np.block([[Ix2, IxIy], 
                  [IxIy, Ix2]])
    
   
    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    #C = np.linalg.det(M) - k*pow(np.trace(M), 2)
    C = Ix2*Iy2 - IxIy**2 - k*((Ix2 + Iy2)**2)
    #print(C.shape) 
    cv.imwrite("C.png", C)

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    

    corners = np.stack(np.where(C > thresh)).T
    print(np.shape(corners))
    return corners, C

