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
    MulA = np.array(design_matrix_gen(4, 2, 2, 'g'))
    A1 = MulA[0]
    A2 = MulA[1]
    Akro = []
    Akro.append(A1)
    Akro.append(A2)
    X = tl.tenalg.kronecker(Akro)
    X = tl.unfold(X, mode=1)
    beta_o = np.array([beta_gen(2, 'g'), beta_gen(2, 'g'),
                       beta_gen(2, 'g'), beta_gen(2, 'g')])
    y = np.matmul(X, beta_o)
    print(X.shape)  # X:2*8*4
    print(beta_o.shape)  # beta_o:4,2
    print(y.shape)  # y:2*8*2
    #[true_gamma, esti_gamma] = run_exp(X, y, [4, 4], 2)
    #ori_gamma_MSE = evaluation(esti_gamma, true_gamma)
    # rp = Merp([4, 4], 2, rand_type='g', target='col', tensor=True, fastQR=True)
    # rp.regenerate_omega()
    # print(len(rp._omegas))
    # print(rp._omegas[0].shape)  # 4,2
    # print(rp._omegas[1].shape)  # 4,2
    # q, r = rp.fastQR(X)
    # print(q.shape)
    # print(r.shape)
    # print(np.matmul(np.transpose(q),q))      #(4,2)
    # print(y.shape)   #(2,2)
    # for i in range(2):
    #     y[i]=np.matmul(q,y[i])  #not transpose
    # y=np.matmul(np.transpose(q),y)
    # gamma, _, _, _ = np.linalg.lstsq(r, y)
    # get the estimate and evaluate it?
