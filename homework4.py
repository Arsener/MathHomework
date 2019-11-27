'''
P190-2
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import *


def Lagrange(x, fx):
    f_A = []
    n = len(x)
    for i in range(n):
        f_A_i = fx[i]
        for j in range(n):
            if i != j:
                f_A_i /= (x[i] - x[j])

        f_A.append(f_A_i)

    def polynomial(new_x):
        ans = 0
        for i in range(n):
            tmp = f_A[i]
            for j in range(n):
                if i != j:
                    tmp *= (new_x - x[j])
            ans += tmp

        return ans

    return polynomial


def solve(A_, b):
    # 增广矩阵
    A = np.concatenate((A_, b), axis=1)
    # print(A)
    size = A.shape[0]

    import copy
    # 消元
    for i in range(size - 1):
        # 找到主元
        k = np.argmax(abs(A[i:size, i]))
        # 交换行
        if k != 0:
            tmp = copy.copy(A[k + i])
            A[k + i] = copy.copy(A[i])
            A[i] = copy.copy(tmp)

        # 作变换
        for j in range(i + 1, size):
            t = A[j, i] / A[i, i]
            A[j, i:] -= A[i, i:] * t

    # 创建一个和b的size相同的矩阵，作为解
    x = np.mat(np.zeros(b.shape), dtype=np.float64)
    # 回代计算
    i = size - 1
    while i >= 0:
        x[i, 0] = (A[i, size] - A[i, i + 1:size] * x[i + 1:, 0]) / A[i, i]
        i -= 1

    return x

def spline_II_edge(x, fx, M_0=0., M_n=0.):
    n = len(x)
    # h
    h = [x[i] - x[i - 1] for i in range(1, n)]
    # mu
    mu = [h[i - 1] / (h[i - 1] + h[i]) for i in range(1, n - 1)]
    # lambda
    lmbd = [1 - m for m in mu]

    # 三对角矩阵
    Mat = np.eye(n-2) * 2
    for i in range(n - 2):
        if i - 1 >= 0:
            Mat[i][i - 1] = mu[i]
        if i + 1 < n - 2:
            Mat[i][i + 1] = lmbd[i]

    # 1阶均差
    f_1_step = [(fx[i] - fx[i - 1]) / (x[i] - x[i - 1]) for i in range(1, n)]
    # 2阶均差 * 6
    f_2_step = [6 * (f_1_step[i - 1] - f_1_step[i - 2]) / (x[i] - x[i - 2]) for i in range(2, n)]
    # 方程式右边的向量
    f_2_step[0] -= mu[0] * M_0
    f_2_step[-1] -= lmbd[-1] * M_n

    # TODO
    M = [M_0] + [i[0] for i in solve(np.mat(Mat), np.mat(f_2_step).T).tolist()] + [M_n]
    print(M)

    def polynomial(new_x):
        if not x[0] <= new_x <= x[-1]:
            raise ValueError("Invalid input! The input is not in the valid region {} to {}.".format(x[0], x[-1]))
        for i in range(n - 1):
            if new_x <= x[i + 1]:
                ans = M[i] * (x[i + 1] - new_x) ** 3 / (6 * h[i])
                ans += M[i + 1] * (new_x - x[i]) ** 3 / (6 * h[i])
                ans += (fx[i] - (M[i] * h[i] * h[i]) / 6) * (x[i + 1] - new_x) / h[i]
                ans += (fx[i + 1] - (M[i + 1] * h[i] * h[i]) / 6) * (new_x - x[i]) / h[i]
                return ans

    return polynomial


def f(x):
    if x <= 1.:
        return 0.5 * x ** 3 - 0.15 * x ** 2 + 0.15 * x
    elif x <= 2:
        return -1.2 * (x - 1) ** 3 + 1.35 * (x - 1) ** 2 + 1.35 * (x - 1) + 0.5
    elif x <= 3:
        return 1.3 * (x - 2) ** 3 - 2.25 * (x - 2) ** 2 + 0.45 * (x - 2) + 2.


if __name__ == '__main__':
    x = [.0, .5, 1., 6., 7., 9.]
    fx = [0., 1.6, 2., 1.5, 1.5, 0.]
    spline = spline_II_edge(x, fx)
    lagrange = Lagrange(x, fx)

    t = [0.01 * i for i in range(901)]

    spline_x = [spline(i) for i in t]
    lagrange_x = [lagrange(i) for i in t]

    plt.plot(x, fx, color='m', linestyle='', marker='o')
    plt.plot(t, lagrange_x, color='b', linestyle='-', marker='', label="Lagrange")
    plt.plot(t, spline_x, color='r', linestyle='-', marker='', label="Spline")
    plt.legend(loc='best')

    plt.show()
