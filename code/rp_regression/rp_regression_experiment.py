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
    rp = Merp(n=tensor_dim, k=k, rand_type='g',
              target='col', tensor=method == 'TRP')
    reduced_X = rp.transform(X)

    true_gamma = solve_linear_regression(X, y)
    esti_gamma = np.matmul(rp._omega, solve_linear_regression(reduced_X, y))
    return [true_gamma, esti_gamma]

    # TODO: do evaluation


def evaluation(beta_hat, beta_true):
    squaredError = []
    error = []
    for i in range(len(beta_true)):
        error.append(beta_true[i] - beta_hat[i])
    for val in error:
        squaredError.append(val * val)
    # MSE
    return sum(squaredError) / len(squaredError)


if __name__ == '__main__':
    X = np.array(design_matrix_gen(3, 2, 2, 'g'))
    beta_o = np.array([beta_gen(1, 'g'), beta_gen(1, 'g')])
    y = np.matmul(X, beta_o)
    print(X[1].shape)
    print(beta_o.shape)
    print(y.shape)
    # for i in range(3):
    #     X.append(design_matrix_gen(3, 2, 2, 'g'))
    #     beta_o.append(beta_gen(2, 'g'))
    #     y.append(np.matmul(X[i], beta_o[i]))
    [true_gamma, esti_gamma] = run_exp(X[0], y[0], [2, 2], 1)
    # gamma_MSE = evaluation(esti_gamma, true_gamma)
    # use x/omega/beta=y get esti beta and calculate MSE of beta?
