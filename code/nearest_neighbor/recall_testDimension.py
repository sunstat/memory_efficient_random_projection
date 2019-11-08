import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors as nn
import logging
from ..merp import Merp
import os
import pickle
from .simulation import normalization, simulate
import matplotlib.pyplot as plt
import tensorly as tl


def test_dimension(iter_steps, smallX,targetNeighbors):
    recallT = []
    recallF = []
    recallTU = []
    recallFU = []
    recallTL = []
    recallFL = []
    for k in range(1200,1205, 5):
        rpb = Merp([128, 128], k, rand_type='g', target='col', tensor=False)
        X_train, X_test = train_test_split(
            smallX, test_size=0.05, train_size=0.95, random_state=23)
        true_neigh = nn(n_neighbors=targetNeighbors)
        true_neigh.fit(X_train, False)
        recall = 0
        recalllist = []
        for i in range(iter_steps):
            rpb.regenerate_omega()
            X_train_rp = rpb.transform(X_train)
            X_test_rp = rpb.transform(X_test)
            # generate predictions
            rp_neigh = nn(n_neighbors=targetNeighbors)
            rp_neigh.fit(X_train_rp)
            # query and calculate recall rate
            true_distances,true_indices = true_neigh.kneighbors(X_test,targetNeighbors)
            pred_distances,pred_indices = rp_neigh.kneighbors(X_test_rp,targetNeighbors)
            
            curr_recall = np.asarray([np.intersect1d(
                true_indices[u], pred_indices[u]).size for u in range(X_test.shape[0])]) / targetNeighbors
            recall = recall+curr_recall.mean()
            recalllist.append(curr_recall.mean())
        recallF.append(recall/iter_steps)
        recallFU.append(np.percentile(recalllist, 97.5))
        recallFL.append(np.percentile(recalllist, 2.5))

    for k in range(1200, 1205, 5):
        rpb = Merp([128, 128], k, rand_type='g', target='col', tensor=True)
        X_train, X_test = train_test_split(
            smallX, test_size=0.05, train_size=0.95, random_state=23)
        true_neigh = nn(n_neighbors=targetNeighbors)
        true_neigh.fit(X_train, False)
        recall = 0
        recalllist = []
        for i in range(iter_steps):
            rpb.regenerate_omega()
            X_train_rp = rpb.transform(X_train)
            X_test_rp = rpb.transform(X_test)
            # generate predictions
            rp_neigh = nn(n_neighbors=targetNeighbors)
            rp_neigh.fit(X_train_rp)
            # query and calculate recall rate
            true_distances,true_indices = true_neigh.kneighbors(X_test,targetNeighbors)
            pred_distances,pred_indices = rp_neigh.kneighbors(X_test_rp,targetNeighbors)
            # print('hello')
            # print(len(true_neighbors))
            # print(len(pred_neighbors))
            # print(len(X_test.shape))
            curr_recall = np.asarray([np.intersect1d(
                true_indices[u], pred_indices[u]).size for u in range(X_test.shape[0])]) / targetNeighbors
            recall = recall+curr_recall.mean()
            recalllist.append(curr_recall.mean())
        recallT.append(recall/iter_steps)
        recallTU.append(np.percentile(recalllist, 97.5))
        recallTL.append(np.percentile(recalllist, 2.5))
    return[recallF, recallFU, recallFL, recallT, recallTU, recallTL]


if __name__ == '__main__':
    mat = scipy.io.loadmat('./data/Flickr_16384.mat')
    X = mat['X']
    print(X.shape)
    ku=100
    X = X[1:ku, :]
    smallX = tl.unfold(X, mode=0)
    smallX = normalization(smallX)
    print(smallX.shape)
    iter_steps = 10
    neighborNum=50

    [recallF, recallFU, recallFL, recallT, recallTU,
        recallTL] = test_dimension(iter_steps, smallX,neighborNum)
    dataSave=[recallF, recallFU, recallFL, recallT, recallTU,recallTL]
    dataSave.append(0)
    dataSave.append(iter_steps)
    dataSave.append(ku)
    dataSave.append(neighborNum)
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    pickle_base = os.path.join(dir_name, 'results')
    pickle.dump(dataSave, open(os.path.join(pickle_base, 'nearneigh_{}.pickle'.format(neighborNum)), 'wb'))
    print(dir_name)
    print(pickle_base)


x = [1200]
y = recallF
z = recallT
yu = recallFU
yl = recallFL
zu = recallTU
zl = recallTL
plt.title('Flickr-16384:Dimension')
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
