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

    q1 = desc1.shape[0]
    q2 = desc2.shape[0]
    distances = np.zeros((q1, q2))
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
        matches = np.vstack((np.arange(q1), np.argmin(distances, axis=1))).T
    elif method == "mutual":
        matches1 = np.vstack((np.arange(q1), np.argmin(distances, axis=1))).T
        #one-way match with image2 as query
        matches2 = np.vstack((np.argmin(distances, axis=0), np.arange(q2))).T
        #mutual match
        matches = np.vstack([x for x in matches1 if x in matches2])

    elif method == "ratio":
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
    else:
        raise NotImplementedError
    return matches

