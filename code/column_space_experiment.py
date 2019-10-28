from tensorsketch.util import square_tensor_gen
from .merp import Merp
import numpy as np
import tensorly as tl

tl.set_backend('numpy')

def exp_column_space(tensor, k, mode=1, method='normal', iteration=100):
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

