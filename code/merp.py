from tensorsketch.random_projection import random_matrix_generator as rm_gen
from tensorsketch.random_projection import tensor_random_matrix_generator as trm_gen
import numpy as np
import tensorly as tl

np.random.seed(233)

# tensor sketch method returning the small matrices before k


def tensor_random_matrix_generator_sp(n_arr, k, typ="g", target='col'):
    """
    routine for usage: A \Omega or \Omega^\top x : n >> m
    :param n_arr: first dimension of random matrix to be generated as a list
    n1*...n_{I+1}*...*n_N
    :param k: second dimension of random matrix to be generated
    :param type:
    :param target:  for column preservation or length preservation,
    for column preservation, we do not need standardization
    :return:
    """
    if not isinstance(n_arr, list):
        raise Exception("type of first parameter must be list")

    types = set(['g', 'u', 'sp0', 'sp1'])
    assert typ in types, "please aset your type of random variable correctly"
    assert target in ['col', 'vec'], "target can only be col or vec"
    Omegas = []
    for n in n_arr:
        if typ == 'g':
            Omega = np.random.normal(0, 1, size=(n, k))
        elif typ == 'u':
            Omega = np.random.uniform(low=-1, high=1, size=(n, k)) * np.sqrt(3)
        elif typ == 'sp0':
            Omega = np.random.choice(
                [-1, 0, 1], size=(n, k), p=[1 / 6, 2 / 3, 1 / 6]) * np.sqrt(3)
        elif typ == 'sp1':
            Omega = np.random.choice([-1, 0, 1], size=(n, k), p=[1 / (2 * np.sqrt(
                n)), 1 - 1 / np.sqrt(n), 1 / (2 * np.sqrt(n))]) * np.sqrt(np.sqrt(n))
        Omegas.append(Omega)
    if target == 'col':
        return Omegas, tl.tenalg.khatri_rao(Omegas)
    return Omegas, tl.tenalg.khatri_rao(Omegas).transpose()/np.sqrt(k)


class Merp(object):

    def __init__(self, n, k, rand_type='g', target='col', tensor=False, fastQR=False):
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
        self._tensor = tensor
        self._fastQR = fastQR
        if fastQR and tensor:
            self._rm_gen_sp = tensor_random_matrix_generator_sp

        # generate omega
        if fastQR and tensor:
            self._omegas, self._omega = self._rm_gen_sp(
                n, k=k, typ=rand_type, target=target)
        else:
            self._omega = self._rm_gen(n, k=k, typ=rand_type, target=target)

    def regenerate_omega(self):
        if self._fastQR and self._tensor:
            self._omegas, self._omega = self._rm_gen_sp(
                self._n, k=self._k, typ=self._type, target=self._target)
        else:
            self._omega = self._rm_gen(
                self._n, k=self._k, typ=self._type, target=self._target)

    def fastQR(self, X):
        '''
        :X: a list 2D matrix, kron(X1, X2, ..., Xn)
        '''
        assert self._fastQR and self._tensor
        assert len(X) == len(self._omegas)
        # qr decompose of Xi\Omega_i
        qs = []
        rs = []
        for i in range(len(X)):
            q, r = np.linalg.qr(np.matmul(X[i], self._omegas[i]))
            qs.append(q)
            rs.append(r)

        q = tl.tenalg.kronecker(qs)
        r = tl.tenalg.khatri_rao(rs)
        return q, r

    def transform(self, X):
        n, d = X.shape
        _n = np.prod(self._n) if isinstance(n, list) else self._n
        assert d == np.prod(_n)
        return np.matmul(X, self._omega)
