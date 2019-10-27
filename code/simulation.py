import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors as nn
import logging
from .merp import Merp

def normalization(X, mode='l2'):
    if not mode in ['gaussian', 'l2']:
        raise ValueError("mode should be 'gaussian' or 'l2'")
    # replace NaN with 0
    X = np.nan_to_num(X)
    if mode == 'gaussian':
        raise NotImplementedError
    elif mode == 'l2':
        # zero center each column
        # then normalize by dividing l2 norm
        X = X - X.mean(axis=0)
        l2 = np.apply_along_axis(np.linalg.norm, 1, X).reshape(-1, 1)
        X = X / l2
        # replace NaN with 0
        X = np.nan_to_num(X)
        return X

def simulate(X, num_iters=10, n_neighbors=10, **merpCfg):
    # normalize input data
    X = normalization(X)
    
    # the merp instance
    rp = Merp(merpCfg)

    # split train and test set
    X_train, X_test = train_test_split(X, test_size=0.05, train_size=0.95, random_state=23)

    # generate ground truth
    true_neigh = nn(n_neighbors=10)
    true_neigh.fit(X_train, return_distance=False)

    recall = []
    for i in range(num_iters):
        rp.regenerate_omega()
        X_train_rp = rp.transform(X_train)
        X_test_rp = rp.transform(X_test)

        # generate predictions
        rp_neigh = nn(n_neighbors=n_neighbors)
        rp_neigh.fit(X_train_rp)

        # query and calculate recall rate
        true_neighbors = true_neigh.kneighbors(X_test)
        pred_neighbors = rp_neigh.kneighbors(X_test_rp)
        curr_recall = np.asarray([np.intersect1d(true_neighbors[i], pred_neighbors[i]).size for i in range(X_test.shape[0])]) / n_neighbors
        recall.append(curr_recall.mean())
    return recall








if __name__ == '__main__':
    mat = scipy.io.loadmat('./data/ImageNet_256.mat')
    print(mat.keys())
    X = mat['X']
    normalized_X = normalization(X)
    print(normalized_X[:10, :])


