import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors as nn
import logging
from .merp import Merp
from .simulation import normalization, simulate
import matplotlib.pyplot as plt
import tensorly as tl
if __name__ == '__main__':
    print('hello')
    mat = scipy.io.loadmat('./data/ImageNet_256.mat')
    recallT = []
    recallF = []
    recallTU = []
    recallFU = []
    recallTL = []
    recallFL = []
    iter_steps = 100
    # print(mat.keys())
    X = mat['X']
    print(X.shape)
    X = X[1:41, :]
    smallX = tl.unfold(X, mode=0)
    smallX = normalization(smallX)
    print(smallX.shape)

    for k in range(10, 65, 5):
        rpb = Merp([16, 16], k, rand_type='g', target='col', tensor=False)
        X_train, X_test = train_test_split(
            smallX, test_size=0.05, train_size=0.95, random_state=23)
        true_neigh = nn(n_neighbors=10)
        true_neigh.fit(X_train, False)
        recall = 0
        recalllist = []
        for i in range(iter_steps):
            rpb.regenerate_omega()
            X_train_rp = rpb.transform(X_train)
            X_test_rp = rpb.transform(X_test)
            # generate predictions
            rp_neigh = nn(n_neighbors=10)
            rp_neigh.fit(X_train_rp)
            # query and calculate recall rate
            true_neighbors = true_neigh.kneighbors(X_test)
            pred_neighbors = rp_neigh.kneighbors(X_test_rp)
            curr_recall = np.asarray([np.intersect1d(
                true_neighbors[i], pred_neighbors[i]).size for i in range(X_test.shape[0])]) / 10
            recall = recall+curr_recall.mean()
            recalllist.append(curr_recall.mean())
        recallF.append(recall/iter_steps)
        recallFU.append(np.percentile(recalllist, 97.5))
        recallFL.append(np.percentile(recalllist, 2.5))
    print(recallF)

    for k in range(10, 65, 5):
        rpb = Merp([16, 16], k, rand_type='g', target='col', tensor=True)
        X_train, X_test = train_test_split(
            smallX, test_size=0.05, train_size=0.95, random_state=23)
        true_neigh = nn(n_neighbors=10)
        true_neigh.fit(X_train, False)
        recall = 0
        recalllist = []
        for i in range(iter_steps):
            rpb.regenerate_omega()
            X_train_rp = rpb.transform(X_train)
            X_test_rp = rpb.transform(X_test)
            # generate predictions
            rp_neigh = nn(n_neighbors=10)
            rp_neigh.fit(X_train_rp)
            # query and calculate recall rate
            true_neighbors = true_neigh.kneighbors(X_test)
            pred_neighbors = rp_neigh.kneighbors(X_test_rp)
            curr_recall = np.asarray([np.intersect1d(
                true_neighbors[i], pred_neighbors[i]).size for i in range(X_test.shape[0])]) / 10
            recall = recall+curr_recall.mean()
            recalllist.append(curr_recall.mean())
        recallT.append(recall/iter_steps)
        recallTU.append(np.percentile(recalllist, 97.5))
        recallTL.append(np.percentile(recalllist, 2.5))
    print(recallT)


x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
y = recallF
z = recallT
yu = recallFU
yl = recallFL
zu = recallTU
zl = recallTL
plt.title('Tensor Test')
plt.plot(x, y, color='r', label='without-tensor', marker='o')
plt.plot(x, yu, color='r', label='without-tensor', linestyle=':', marker='o')
plt.plot(x, yl, color='r', label='without-tensor', linestyle=':', marker='o')
plt.plot(x, z, color='b', label='with-tensor', marker='v')
plt.plot(x, zu, color='b', label='with-tensor', linestyle=':', marker='v')
plt.plot(x, zl, color='b', label='with-tensor', linestyle=':', marker='v')
plt.legend()
plt.xlabel("reduced dimension")
plt.ylabel("recall")
plt.show()
