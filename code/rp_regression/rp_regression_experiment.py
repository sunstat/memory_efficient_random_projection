from ..merp import Merp
import numpy as np

'''
min ||Y - X\Omega\beta||_2
'''

def solve_linear_regression(X, y):
    '''
    return argmin_gamma (||Y - X\gamma||_2)^2
    '''
    gamma, _, _, _ = np.linalg.lstsq(X, y)
    return gamma

def design_matrix_gen(order, n, d, typ='g'):
    Ai = []
    if typ == 'g':
        for _ in range(order):
            Ai.append(np.random.normal(0.0, 1.0, size=(n, d)))
    else:
        raise NotImplementedError
    return Ai

def beta_gen(d, typ='g'):
    if type == 'g':
        beta = np.random.normal(0.0, 1.0, size=(d,))
    elif type == 'u':
        beta = np.random.uniform(-1.0, 1.0, size=(d,))
    else:
        raise NotImplementedError
    return beta

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
def evaluation(beta_hat,beta_true):
    absError = []
    squaredError = []
    error = []
    for i in range(len(beta_true)):
        error.append(beta_true[i] - beta_hat[i])
    for val in error:
        squaredError.append(val * val)
    #MSE
    return sum(squaredError) / len(squaredError)


    



    

