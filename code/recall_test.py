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


    for k in range(10,65,5):
        rpb = Merp([16,16], k, rand_type='g', target='col', tensor=False)
        X_train, X_test = train_test_split(smallX, test_size=0.05, train_size=0.95, random_state=23)
        true_neigh = nn(n_neighbors=10)
        true_neigh.fit(X_train, False)
        recall = 0
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
            recall=recall+curr_recall.mean()
        recallF.append(recall/10)
    print(recallF)



    for k in range(10,65,5):
        rpb = Merp([16,16], k, rand_type='g', target='col', tensor=True)
        X_train, X_test = train_test_split(smallX, test_size=0.05, train_size=0.95, random_state=23)
        true_neigh = nn(n_neighbors=10)
        true_neigh.fit(X_train, False)
        recall = 0
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
            recall=recall+curr_recall.mean()
        recallT.append(recall/10)
    print(recallT)





x = [10,15,20,25,30,35,40,45,50,55,60]
y=recallF
z=recallT
plt.title('Tensor Test')
plt.plot(x,y,label='without-tensor',marker='o')
plt.plot(x,z,label='with-tensor',marker='v')
plt.legend()
plt.xlabel("reduced dimension")
plt.ylabel("recall")
plt.show()




    