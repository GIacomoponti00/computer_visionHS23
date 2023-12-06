

# This function should calculate the normalized histogram of RGB colors occurring within the bound-
# ing box defined by (xmin, xmax) (ymin, ymax) within the current video frame. The histogram is
# obtained by binning each color channel (R,G,B) into hist_bin bins. Consequently, we essentially
# assume only hist bin possible distinct colors instead of the full 256 possible colors in 24-bit color
# space. You may store the histogram using whichever data structure you prefer, as it is (almost)
# only used by the functions you will write.
import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin) :
    
    #get the image in the bounding box
    bbox = frame[ymin:ymax, xmin:xmax, :]
    
    #get the histogram of the image
    hist, bins = np.histogram(bbox, bins=hist_bin)
    
    return hist