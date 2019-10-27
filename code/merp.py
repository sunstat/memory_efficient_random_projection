from tensorsketch.random_projection import random_matrix_generator as rm_gen
from tensorsketch.random_projection import tensor_random_matrix_generator as trm_gen
import numpy as np
        
np.random.seed(233)

class Merp(object):

    def __init__(self, n, k, rand_type='g', target='col', tensor=False):
        '''
        :param: n, first dimension of random matrix to be generated
        :param: k, second dimension of random matrix to be generated
        :param: rand_type, random distribution
        :param: target
        '''
        self._n = n
        self._k = k
        self._type = rand_type
        self._target = target
        self._rm_gen = trm_gen if tensor else rm_gen
        
        # generate omega
        self._omega = self._rm_gen(n=n, k=k, typ=rand_type, target=target)

    def regenerate_omega(self):
        self._omega = self._rm_gen(n=self._n, 
                                   k=self._k, 
                                   typ=self._rand_type, 
                                   target=self._target)
    
    def transform(self, X):
        n, d = X.shape
        _n = np.prod(self._n) if isinstance(n, list) else self._n
        assert d == np.prod(_n)
        return np.matmul(X, self._omega)