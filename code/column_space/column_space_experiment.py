import os
from tensorsketch.util import square_tensor_gen
from ..merp import Merp
import numpy as np
import tensorly as tl
import pickle
# import hdmedians as hd
from .plot import plot

def orthogonal_matrix_gen(dim1, dim2):
    H = np.random.randn(dim1, dim2)
    Q, _ = np.linalg.qr(H)
    #print(Q)
    return Q

def average_mat(Xs, method='average'):
    # if method == 'geomedian':
    #     X = np.asarray([Xs[i].reshape(-1,) for i in range(len(Xs))])
        #print(X.shape)
        # geomedian = np.array(hd.geomedian(X, axis=0))
        # avg = geomedian.reshape(Xs[0].shape)
    # elif method == 'average':
    avg = np.sum(Xs, axis=0) / len(Xs)
    # else:
    #     raise NotImplementedError
    return avg

def run_exp(tensor, k, mode=1, method='normal', iteration=100, var_reduction='geomedian'):
    '''
    :tensor: generated tensor
    :k: reduced dimension
    :iteration: total run of iterations
    '''
    assert method in ['TRP', 'normal']
    assert var_reduction in ['average', 'geomedian', None]
    shape = list(tensor.shape)
    del shape[mode]

    if method == 'TRP':
        rp = Merp(n=shape, k=k, tensor=True)
    else:
        rp = Merp(n=shape, k=k, tensor=False)
    
    # unfold tensor
    X = tl.unfold(tensor, mode=1)
    relative_errs = []
    for _ in range(iteration):
        if var_reduction is None:
            rp.regenerate_omega()
            reduced_X = rp.transform(X)
        
            # qr decomposition
            q, _ = np.linalg.qr(reduced_X, mode='reduced')
        
            # calculate relative error
            err = np.linalg.norm(np.matmul(np.matmul(q, q.T), X)-X, ord='fro')
            del q
        else:
            X_hats = []
            for _ in range(5):
                rp.regenerate_omega()
                reduced_X = rp.transform(X)
                # qr decomposition
                q, _ = np.linalg.qr(reduced_X, mode='reduced')
                X_hats.append(np.matmul(np.matmul(q, q.T), X))
                
                del q
            
            avg_X_hat = average_mat(X_hats, var_reduction)
            del X_hats
            # calculate relative error
            err = np.linalg.norm(avg_X_hat-X, ord='fro')
        
        relative_err = err / np.linalg.norm(X, ord='fro')
        relative_errs.append(relative_err)
    return sorted(relative_errs)

def dense_tensor_gen(dim, r=10, mode=1, k=40):
    m, n = dim[mode], np.prod(dim) // dim[mode]
    assert m >= k
    # generate sigma
    sigma = np.diag([1] * r + [1/np.power(2, i) for i in range(1, k-r+1)])
    
    U = orthogonal_matrix_gen(m, k)
    V = orthogonal_matrix_gen(n, k)
    X = np.matmul(np.matmul(U, sigma), V.T)
    return tl.tensor(X.reshape(dim))

def column_space_experiment(tensor_dim=[(500, 500)], k=[5, 10, 15, 20, 25], mode=1, iteration=100):
    print('\tStart column space experiment...')
    # result dictionary
    res = dict()
    for dim in tensor_dim:
        res[dim] = dict()
        for reduced_k in k:
            res[dim][reduced_k] = dict()
            for var_red in [None, 'average', 'geomedian']:
                curr_key = 'raw' if var_red is None else var_red
                res[dim][reduced_k][curr_key] = dict()
                # generate tensor
                # 1. dense tensor
                dense_tensor = dense_tensor_gen(dim, r=10, mode=mode) #tl.tensor(np.random.normal(loc=0.0, scale=1.0, size=dim))
                # 2. square tensor
                lr_tensor, _ = square_tensor_gen(n=dim[mode], r=5, dim=len(dim))
                for tensor_gen in ['dense_tensor', 'lr_tensor']:
                    res[dim][reduced_k][curr_key][tensor_gen] = dict()
                    for method in ['TRP', 'normal']:
                        tensor = dense_tensor if tensor_gen == 'dense_tensor' else lr_tensor
                        relative_err = run_exp(tensor=tensor, k=reduced_k, mode=mode, method=method, iteration=iteration, var_reduction=var_red)
                        # store results
                        res[dim][reduced_k][curr_key][tensor_gen][method] = relative_err
                del dense_tensor
                del lr_tensor
    print('\tFinished column space experiment...')
    return res

if __name__ == '__main__':
    
    tl.set_backend('numpy')
    #tensor_dim = (500, 500)
    tensor_dim = (200, 200, 200)
    res = column_space_experiment(tensor_dim=[tensor_dim], k=[5, 10, 15, 20, 25], mode=1, iteration=100)
    
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    pickle_base = os.path.join(dir_name, 'results')
    # save pickle
    pickle.dump(res, open(os.path.join(pickle_base, 'tensor_dim_{}.pickle'.format(tensor_dim)), 'wb'))
    plot(tensor_dim, res=res, fig_name='3D-(200×200×200) Tensor Sketching')
    



