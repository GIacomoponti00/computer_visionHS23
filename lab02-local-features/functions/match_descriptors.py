import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    q1 = desc1.shape[0]
    q2 = desc2.shape[0]
    distances = np.zeros((q1, q2))
    #print("ssd")
    for i in range(q1):
        for j in range(q2):
            distances[i,j] = np.sum((desc1[i,:] - desc2[j,:])**2)
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis

        matches = np.vstack((np.arange(q1), np.argmin(distances, axis=1))).T


        #raise NotImplementedError
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        #one-way match
        matches1 = np.vstack((np.arange(q1), np.argmin(distances, axis=1))).T
        #one-way match with image2 as query
        matches2 = np.vstack((np.argmin(distances, axis=0), np.arange(q2))).T
        #mutual match
        matches = np.vstack([x for x in matches1 if x in matches2])

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        #raise NotImplementedError
        q1, q2 = desc1.shape[0], desc2.shape[0]
        matches = np.zeros((q1, 2))

        for i in range(q1):
            distances_i = distances[i,:]
            min_dist = np.min(distances_i)
            second_min_dist = np.partition(distances_i, 1, axis=0)[1]
            if (min_dist / second_min_dist) < ratio_thresh:
                matches[i,:] = np.array([i, np.argmin(distances_i)])
        matches = matches[matches[:,0] != 0, :]
        matches = matches.astype(int)
        #
        #matches = np.vstack((np.arange(q1), np.argmin(distances, axis=1)))
        #matches = matches[matches[:,0] != 0, :]
        #matches = matches.astype(int)

        #
        #matches
        #raise NotImplementedError
    else:
        raise NotImplementedError
    return matches

