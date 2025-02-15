from ..merp import Merp
import numpy as np
import tensorly as tl


'''
min ||Y - X\Omega\beta||_2
'''


def solve_linear_regression(X, y):
    '''
    return argmin_gamma (||Y - X\gamma||_2)^2
    '''
    gamma, _, _, _ = np.linalg.lstsq(X, y)
    return gamma


def design_matrix_gen(order, n, d, type='g'):
    Ai = []
    if type == 'g':
        for _ in range(order):
            Ai.append(np.random.normal(0.0, 1.0, size=(n, d)))
    else:
        raise NotImplementedError
    return Ai


def beta_gen(d, type='g'):
    if type == 'g':
        beta = np.random.normal(0.0, 1.0, size=(d, 1))
    elif type == 'u':
        beta = np.random.uniform(-1.0, 1.0, size=(d, 1))
    else:
        raise NotImplementedError
    return beta


def y_gen(X, beta, noise_level=0.1):
    print(X.shape)
    print(beta.shape)
    y = X @ beta + np.random.normal(0.0, noise_level, size=(X.shape[0], 1))
    return y


def run_exp(A, X, y, beta, k, tensor_dim, method='TRP', target='vec', fastQR=False, iteration=100):
    n, m = X.shape
    assert method in ['TRP', 'normal']
    assert target == 'vec'

    relative_errs = []
    residuals = []
    rp = Merp(n=tensor_dim, k=k, rand_type='g',
              target=target, tensor=(method=='TRP'), fastQR=(method=='TRP' and fastQR))
    for _ in range(iteration):
        rp.regenerate_omega()
        if fastQR:
            raise NotImplementedError
            '''
            if method == 'TRP':
                q, r = rp.fastQR(A)
            else:
                reduced_X = rp.transform(X)
                q, r = np.linalg.qr(rp.transform(X))
            est_beta, residual, _, _ = np.linalg.lstsq(r, q.T @ y)
            err = np.linalg.norm((rp._omega @ est_beta)-beta, ord=2)
            relative_err = err / np.linalg.norm(beta, ord=2)
            relative_errs.append(relative_err)
            residuals.append(residual)
            '''
        else:
            reduced_X = rp.transform(X, right=(target=='col'))
            reduced_y = rp.transform(y, right=(target=='col'))
            est_beta, residual, _, _ = np.linalg.lstsq(reduced_X, reduced_y, rcond=None)
            print('beta:', beta)
            print('est_beta:', est_beta)
            err = np.linalg.norm(est_beta - beta, ord=2)
            relative_err = err / np.linalg.norm(beta, ord=2)
            relative_errs.append(relative_err)
            residuals.append(residual)

    return relative_errs, residuals


def evaluation(beta_hat, beta_true):
    squaredError = []
    error = []
    for i in range(len(beta_true)):
        error.append(beta_true[i] - beta_hat[i])
    for val in error:
        squaredError.append(val * val)
    # MSE
    return sum(squaredError) / len(squaredError)

def simple_row_KR(A,B):
    assert A.shape[0]==B.shape[0]
    C=np.array([[]])
    for i in range(A.shape[0]):
        C[i:]=tl.tenalg.kronecker(A[i:],B[i:])
    return C


def rp_regression_experiment(A_dim=(50, 20), order=2, k=[5, 10, 15, 20, 25], noise_level=0.1, iteration=100):
    '''
    estimate min||Y-AX||_2 by min||Omega Y - Omega AX||
    '''
    A = design_matrix_gen(order, A_dim[0], A_dim[1])
    dim = [A_dim[0] for _ in range(order)]
    print('dim:', dim)
    # generate beta, y
    X = tl.tenalg.kronecker(A)
    beta = beta_gen(X.shape[1])
    y = y_gen(X, beta, noise_level=noise_level)
    res = dict()
    for reduced_k in k:
        res[reduced_k] = dict()
        for method in ['TRP', 'normal']:
            res[reduced_k][method] = dict()
            relative_err, residual = run_exp(
                A, X, y, beta, reduced_k, dim, method=method, iteration=iteration)
            res[reduced_k][method]['relative_err'] = relative_err
            res[reduced_k][method]['residual'] = residual
    return res



#testfunc below
def new_rp_regression_test_simulation(A_dim=(20, 30), order=2, k=[5, 10, 15, 20, 25], noise_level=0.1, iteration=100):
    A = design_matrix_gen(order, A_dim[0], A_dim[1])
    dim = [A_dim[1] for _ in range(order)]
    print(dim)
    # generate beta, y
    beta = beta_gen(np.prod(dim))
    X = tl.tenalg.kronecker(A)
    y = y_gen(X, beta, noise_level=noise_level)
    rp=Merp([4, 4], 2, rand_type='g', target='col', tensor=True, fastQR=True)
    rp.regenerate_omega()
    q,r=rp.fastQR(np.matmul(rp._omegas,X))
    qt=np.transpose(q)
    beta_hat=np.linalg.lstsq(np.matmul(np.matmul(qt,q), x),np.matmul(rp._omegas,y))
    #get the analysis for beta_hat and beta
    
    


if __name__ == '__main__':
    A_dim = (10, 5)
    order = 2
    k = [26]
    noise = 0.1
    iteration = 1
    res = rp_regression_experiment(A_dim=A_dim, 
                                   order=order, 
                                   k=k, 
                                   noise_level=noise,
                                   iteration=iteration)
    print(res)


