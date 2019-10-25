'''
P91-1
'''

import numpy as np


# 将矩阵A拆分成A = D - L - U
def divide_matrix(H):
    n = H.shape[0]
    D = np.mat([[H[i, j] if i == j else 0 for j in range(n)] for i in range(n)])
    L = np.mat([[-H[i, j] if i > j else 0 for j in range(n)] for i in range(n)])
    U = np.mat([[-H[i, j] if i < j else 0 for j in range(n)] for i in range(n)])
    return D, L, U


# 生成n*n的Hilbert矩阵
def hilbert_martrix(n):
    H = [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, 1 + n)]
    return np.mat(H)


# Jacobi迭代法
def jacobi(H, b, k, x=None):
    # 初始向量为零向量
    if x is None:
        x = np.zeros(b.shape)

    D, L, U = divide_matrix(H)
    D_i = D.I
    Bj = D_i * (L + U)
    f = D_i * b
    for i in range(k):
        x = Bj * x + f

    return x


# SOR迭代法
def sor(H, b, k, w=1., x=None):
    # 初始向量为零向量
    if x is None:
        x = np.zeros(b.shape)

    D, L, U = divide_matrix(H)
    D_i = (D - w * L).I
    Lw = D_i * ((1 - w) * D + w * U)
    f = w * D_i * b
    for i in range(k):
        x = Lw * x + f

    return x


if __name__ == '__main__':
    n = 6
    H = hilbert_martrix(n)
    print('Hilbert Matrix:\n{}'.format(H))
    x = np.ones((H.shape[0], 1))
    b = H * x
    x_j = jacobi(H, b, 10)
    print('Jacobi:\n{}'.format(x_j))
    x_s = sor(H, b, 1000, w=1.)
    print('SOR:\n{}'.format(x_s))
    print('Infinite norm of (x - x_s):')
    print(np.linalg.norm(x - x_s, ord=np.inf))
