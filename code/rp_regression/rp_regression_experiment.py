from ..merp import Merp
import numpy as np

def solve_linear_regression(X, y):
    '''
    return argmin_gamma (||Y - X\gamma||_2)^2
    '''
    gamma, _, _, _ = np.linalg.lstsq(X, y)
    return gamma

def run_exp(X, y, tensor_dim, k, mode=0, method='TRP'):
    n, m = X.shape
    assert n == tensor_dim[mode]
    assert np.prod(tensor_dim) == n * m
    assert method in ['TRP', 'normal']

    del tensor_dim[mode]
    rp = Merp(n=tensor_dim, k=k, rand_type='g', target='col', tensor=method=='TRP')
    reduced_X = rp.transform(X)

    true_gamma = solve_linear_regression(X, y)
    esti_gamma = np.matmul(rp._omega, solve_linear_regression(reduced_X, y))

    #TODO: do evaluation




    

