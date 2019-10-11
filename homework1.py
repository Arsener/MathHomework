import numpy as np
import copy


def solve(A_, b):
    # 增广矩阵
    A = np.concatenate((A_, b), axis=1)
    # print(A)
    size = A.shape[0]

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


def get_info(A, b):
    # 输出A
    print('A:')
    print(A)
    # 输出b
    print('b:')
    print(b)
    # 输出det A
    print('det A:')
    print(np.linalg.det(A))
    # 输出解向量x
    print('x:')
    print(solve(A, b))
    # 输出A的条件数
    print('cond(A)_1:')
    print(np.linalg.cond(A, p=1))
    print('cond(A)_2:')
    print(np.linalg.cond(A, p=2))
    print('cond(A)_inf:')
    print(np.linalg.cond(A, p=np.inf))
    print('\n')


if __name__ == '__main__':
    print('Problem (1):')
    A1 = np.mat([[3.01, 6.03, 1.99],
                 [1.27, 4.16, -1.23],
                 [0.987, -4.81, 9.34]])
    b1 = np.mat([1.0, 1.0, 1.0]).T
    get_info(A1, b1)

    print('Problem (2):')
    A2 = np.mat([[3.00, 6.03, 1.99],
                 [1.27, 4.16, -1.23],
                 [0.990, -4.81, 9.34]])
    b2 = np.mat([1.0, 1.0, 1.0]).T
    get_info(A2, b2)
