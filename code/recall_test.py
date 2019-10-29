import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors as nn
import logging
from .merp import Merp
from .simulation import normalization,simulate
import matplotlib.pyplot as plt
import tensorly as tl
if __name__ == '__main__':
    print('hello')
    mat = scipy.io.loadmat('./data/ImageNet_256.mat')
    recallT=[]
    recallF=[]
    # print(mat.keys())
    X = mat['X']
    print(X.shape)
    X=X[1:41,:]
    smallX=tl.unfold(X,mode=0)
    smallX = normalization(smallX)
    print(smallX.shape)
    rpa = Merp([16,16], 20, rand_type='g', target='col', tensor=False)
    X_train, X_test = train_test_split(smallX, test_size=0.05, train_size=0.95, random_state=23)
    true_neigh = nn(n_neighbors=10)
    true_neigh.fit(X_train, False)
    recall = []
    for i in range(10):
        rpa.regenerate_omega()
        X_train_rp = rpa.transform(X_train)
        X_test_rp = rpa.transform(X_test)

        # generate predictions
        rp_neigh = nn(n_neighbors=10)
        rp_neigh.fit(X_train_rp)

        # query and calculate recall rate
        true_neighbors = true_neigh.kneighbors(X_test)
        pred_neighbors = rp_neigh.kneighbors(X_test_rp)
        curr_recall = np.asarray([np.intersect1d(true_neighbors[i], pred_neighbors[i]).size for i in range(X_test.shape[0])]) / 10
        recall.append(curr_recall.mean())
    print(recall)
    recallF=recall

    rpb = Merp([4,64], 20, rand_type='g', target='col', tensor=True)
    X_train, X_test = train_test_split(smallX, test_size=0.05, train_size=0.95, random_state=23)
    true_neigh = nn(n_neighbors=10)
    true_neigh.fit(X_train, False)
    recall = []
    for i in range(10):
        rpb.regenerate_omega()
        X_train_rp = rpb.transform(X_train)
        X_test_rp = rpb.transform(X_test)

        # generate predictions
        rp_neigh = nn(n_neighbors=10)
        rp_neigh.fit(X_train_rp)

        # query and calculate recall rate
        true_neighbors = true_neigh.kneighbors(X_test)
        pred_neighbors = rp_neigh.kneighbors(X_test_rp)
        curr_recall = np.asarray([np.intersect1d(true_neighbors[i], pred_neighbors[i]).size for i in range(X_test.shape[0])]) / 10
        recall.append(curr_recall.mean())
    print(recall)
    recallT=recall
x = [1,2,3,4,5,6,7,8,9,10]
y=recallF
z=recallT
plt.title('Tensor Test')
plt.plot(x,y,label='without-tensor')
plt.plot(x,z,label='with-tensor')
plt.legend()
plt.xlabel("random steps")
plt.ylabel("recall")
plt.show()




    