from tensorsketch.util import square_tensor_gen
from .merp import Merp
import numpy as np
import tensorly as tl
import pickle

tl.set_backend('numpy')

def run_exp(tensor, k, mode=1, method='normal', iteration=100):
    '''
    :tensor: generated tensor
    :k: reduced dimension
    :iteration: total run of iterations
    '''
    assert method in ['TRP', 'normal']
    # unfold tensor
    X = tl.unfold(tensor=tensor, mode=mode)
    n, d = X.shape

    if method == 'TRP':
        rp = Merp(n=d, k=k, tensor=True)
    else:
        rp = Merp(n=d, k=k, tensor=False)
    
    relative_errs = []
    for i in range(iteration):
        rp.regenerate_omega()
        reduced_X = rp.transform(X)
        
        # qr decomposition
        q, _ = np.linalg.qr(reduced_X, mode='complete')
        
        # calculate relative error
        err = np.linalg.norm(np.matmul(np.matmul(q, q.T), X)-X, ord='fro')
        relative_err = err / np.linalg.norm(X, ord='fro')
        relative_errs.append(relative_err)
    return relative_errs


def column_space_experiment(tensor_dim=[(900, 900), (400, 400, 400), (100, 100, 100, 100)], k=[5, 10, 15, 20, 25], mode=1, iteration=100):
    # result dictionary
    res = dict()
    for dim in tensor_dim:
        for reduced_k in k:
            curr_res = dict()
            # generate tensor
            # 1. dense tensor
            dense_tensor = tl.tensor(np.random.normal(loc=0.0, scale=1.0, size=dim))
            # 2. square tensor
            lr_tensor, _ = square_tensor_gen(n=dim[0], r=5, dim=len(dim))
            for tensor_gen in ['dense_tensor', 'lr_tensor']:
                for method in ['TRP', 'normal']:
                    tensor = dense_tensor if tensor_gen == 'dense_tensor' else lr_tensor
                    relative_err = run_exp(tensor=tensor, k=reduced_k, mode=mode, method=method, iteration=iteration)
                    # store results
                    curr_res[(tensor_gen, method)] = relative_err
            res[(dim, reduced_k)] = curr_res
    return res



