'''
P91-1
'''

import numpy as np


def divide_matrix(H):
    n = H.shape[0]
    D = np.mat([[H[i, j] if i == j else 0 for j in range(n)] for i in range(n)])
    L = np.mat([[-H[i, j] if i > j else 0 for j in range(n)] for i in range(n)])
    U = np.mat([[-H[i, j] if i < j else 0 for j in range(n)] for i in range(n)])
    return D, L, U


def hilbert_martrix(n):
    H = [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, 1 + n)]
    return np.mat(H)


def jacobi(H, b, k, x=None):
    if x is None:
        x = np.zeros(b.shape)

    D, L, U = divide_matrix(H)
    D_i = D.I
    Bj = D_i * (L + U)
    f = D_i * b
    for i in range(k):
        x = Bj * x + f

    return x


def sor(H, b, k, w=1., x=None):
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
    H = hilbert_martrix(6)
    print(H)
    b = np.sum(H, axis=1)
    x = jacobi(H, b, 5)
    print(x)
    x = sor(H, b, 15000)
    print(x)
