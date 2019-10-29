from tensorsketch.util import square_tensor_gen
from ..merp import Merp
import numpy as np
import tensorly as tl
import pickle
import hdmedians as hd

def orthogonal_matrix_gen(dim1, dim2):
    H = np.random.randn(dim1, dim2)
    Q, _ = np.linalg.qr(H)
    print(Q)
    return Q

def average_mat(Xs, method='geomedian'):
    if method == 'geomedian':
        X = np.asarray([Xs[i].reshape(-1,) for i in range(len(Xs))])
        print(X.shape)
        geomedian = np.array(hd.geomedian(X, axis=0))
        avg = geomedian.reshape(Xs[0].shape)
    elif method == 'average':
        avg = np.sum(Xs, axis=0) / len(Xs)
    else:
        raise NotImplementedError
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
        else:
            X_hats = []
            for _ in range(5):
                rp.regenerate_omega()
                reduced_X = rp.transform(X)
                # qr decomposition
                q, _ = np.linalg.qr(reduced_X, mode='reduced')
                X_hats.append(np.matmul(np.matmul(q, q.T), X))
            
            avg_X_hat = average_mat(X_hats, var_reduction)
            
            # calculate relative error
            err = np.linalg.norm(avg_X_hat-X, ord='fro')
        
        relative_err = err / np.linalg.norm(X, ord='fro')
        relative_errs.append(relative_err)
    return sorted(relative_errs)

def dense_tensor_gen(dim, r=10, mode=1, k=40):
    m, n = dim[mode], np.prod(dim) // dim[mode]
    assert m >= k
    # generate sigma
    print([1/np.power(2, i) for i in range(1, k-r+1)])
    sigma = np.diag([1] * r + [1/np.power(2, i) for i in range(1, k-r+1)])
    
    U = orthogonal_matrix_gen(m, k)
    V = orthogonal_matrix_gen(n, k)
    X = np.matmul(np.matmul(U, sigma), V.T)
    #print('X:', X)
    return tl.tensor(X.reshape(dim))

def column_space_experiment(tensor_dim=[(900, 900), (400, 400, 400), (100, 100, 100, 100)], k=[5, 10, 15, 20, 25], mode=1, iteration=100):
    print('\tStart column space experiment...')
    # result dictionary
    res = dict()
    for dim in tensor_dim:
        for reduced_k in k:
            for var_red in [None, 'geomedian', 'average']:
                curr_res = dict()
                # generate tensor
                # 1. dense tensor
                dense_tensor = dense_tensor_gen(dim, r=10, mode=mode) #tl.tensor(np.random.normal(loc=0.0, scale=1.0, size=dim))
                # 2. square tensor
                lr_tensor, _ = square_tensor_gen(n=dim[mode], r=5, dim=len(dim))
                for tensor_gen in ['dense_tensor', 'lr_tensor']:
                    for method in ['TRP', 'normal']:
                        tensor = dense_tensor if tensor_gen == 'dense_tensor' else lr_tensor
                        relative_err = run_exp(tensor=tensor, k=reduced_k, mode=mode, method=method, iteration=iteration)
                        # store results
                        curr_res[(tensor_gen, method)] = relative_err
                res[(dim, reduced_k, var_red)] = curr_res
    print('\tFinished column space experiment...')
    return res

if __name__ == '__main__':
    
    tl.set_backend('numpy')
    tensor_dim = (900, 900)
    res = column_space_experiment(tensor_dim=[tensor_dim], k=[5, 10, 15, 20, 25], mode=1, iteration=100)
    #for cfg in res.keys():
    #    print(cfg, ':', res[cfg])
    



