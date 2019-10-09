import numpy as np
import copy


def solve(A_, b):
    # 增广矩阵
    A = np.concatenate((A_, b), axis=1)
    # print(A)
    size = A.shape[0]

    for i in range(size - 1):
        # 找到主元
        k = np.argwhere(abs(A[i:size, i]) == max(abs(A[i:size, i]))[0, 0])[0, 0]
        print(k)
        # 交换行
        if k != 0:
            tmp = copy.copy(A[k + i])
            A[k + i] = copy.copy(A[i])
            A[i] = copy.copy(tmp)

        for j in range(i + 1, size):
            t = A[j, i] / A[i, i]
            A[j, i:] -= A[i, i:] * t

            # print(A)

    x = np.ma
    pass


if __name__ == '__main__':
    A1 = np.mat([[3.01, 6.03, 1.99],
                 [1.27, 4.16, -1.23],
                 [0.987, -4.81, 9.34]])
    b1 = np.mat([1, 1, 1]).T
    print(np.linalg.det(A1))
    solve(A1, b1)

    A2 = np.mat([[3.00, 6.03, 1.99],
                 [1.27, 4.16, -1.23],
                 [0.990, -4.81, 9.34]])
    b2 = np.mat([1, 1, 1]).T

# a = np.mat([[1, 3], [4, 2]])
# print(a)
# print(a.T)
# print(np.linalg.det(a))
