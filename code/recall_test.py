import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors as nn
import logging
from .merp import Merp
from .simulation import normalization,simulate
import matplotlib as plt
import tensorly as tl
if __name__ == '__main__':
    print('hello')
    mat = scipy.io.loadmat('./data/ImageNet_256.mat')
    # print(mat.keys())
    X = mat['X']
    print(X.shape)
    smallX=tl.unfold(X,mode=1)
    print(smallX.shape)
    smallX = normalization(smallX)
    rp = Merp(16, 16, rand_type='g', target='col', tensor=False)
    X_train, X_test = train_test_split(X, test_size=0.05, train_size=0.95, random_state=23)
    true_neigh = nn(n_neighbors=10)
    true_neigh.fit(X_train, False)
    recall = []
    for i in range(10):
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
    print(recall)