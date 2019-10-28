import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
import pickle
from sklearn import cluster
from sklearn.cluster import MiniBatchKMeans
batch_size = 1000


def getImage(str):

    mat = io.ImageCollection(str)

    return mat


def getOrb(mat):
    label = {}
    orbdic = []
    siftt = cv2.xfeatures2d.SIFT_create()
    for i in range(len(mat)):
        kp = siftt.detect(mat[i], None)
        des = siftt.compute(mat[i], kp)
        # des_pca=(PCA(n_components=128).fit_transform(des[1].transpose())).transpose()
        # des[1]=Vlad(des[1])
        # print des[1].shape
        # final=Vlad(des[1])
        final = des[1]
        print(i)
        print(final.shape)
        # print final
        orbdic.append(final)

        # label[i]=des[1].shape[0]
    return orbdic, label
#################################################################################
# use minibatchkmeans to train your sift descriptors,then you will get a codebook with k words


def codebook(orbdic):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=64, max_iter=1000, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0).fit(orbdic)
    label = mbk.labels_
    centroid = mbk.cluster_centers_
    return label, centroid
#########################################################################################

###########################################################################################
# assign all the sift descriptors of each picture to the nearest codeword,and we can use a K*128 vlad vector to descrip each
# picture
# refer to my blog
# notice:if picture x doesn't have any sift descriptor which belongs to code word i ,we use a 128d zero vector to represent it.


def vlad(locdpt, centroid):
    dis = {}
    final = []
    for i in range(locdpt.shape[0]):
        des = []
        for j in range(centroid.shape[0]):
            des.append(np.linalg.norm(locdpt[i]-centroid[j]))
        if dis.has_key(np.argmin(des)):
            dis[np.argmin(des)].append(locdpt[i])
        else:
            dis[np.argmin(des)] = [locdpt[i]]
    # print len(dis),dis.keys()
    for i in range(64):
        total = 0
        if dis.has_key(i):
            for j in range(len(dis[i])):

                total = total+dis[i][j]-centroid[i]
            if np.linalg.norm(total) != 0:
                total = total/np.linalg.norm(total)

        else:
            total = np.zeros((128,))
        final.append(total)
    print(len(final))
    final = concate2(final, len(final))
    return final
###############################################################################################


def gfinal(mat):
    gf = []
    orbdic, label = getOrb(mat)
    database = concate(orbdic, len(orbdic))
    label, centroid = codebook(database)
    print(centroid.shape)
    for i in range(len(orbdic)):
        gf.append(vlad(orbdic[i], centroid))
    return gf


def concate2(orbdic, l):
    # print "concate-all-features-vector"
    database = orbdic[0]
    for i in range(1, l):
        # print orbdic[i].shape
        database = np.hstack((database, orbdic[i]))
    return database


def concate(orbdic, l):
    # print "concate-all-features-vector"
    database = orbdic[0]
    for i in range(1, l):
        database = np.vstack((database, orbdic[i]))
    return database


def train(database):
    label, centroid = codebook(database)
    print(centroid.shape)
    with open("codebook.pickle", 'wb')as pk:
        pickle.dump(centroid, pk)


def test(orbdic):
    final = []
    with open("codebook.pickle", 'rb')as pk:
        codebook = pickle.load(pk)
    print(codebook.shape)
    for i in range(len(orbdic)):
        final.append(vlad(orbdic[i], codebook))
    final = concate(final, len(final))
    return final


if __name__ == "__main__":
    # the picture path
    filename = "klboat"
    str = filename+"/*.jpg"
    # get each picture's sift-descriptor
    mat = getImage(str)
    io.imshow(mat[0])
    orbdic, label = getOrb(mat)
    database = concate(orbdic, len(orbdic))
    # train for codebook
    train(database)
    # do NN-assign
    final = test(orbdic)
    # save vlad descriptor,size 86*(k*128)
    print(final.shape)
    with open("db.pickle", 'wb')as pk:
        pickle.dump(final, pk)
