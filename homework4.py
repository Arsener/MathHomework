'''
P190-2
'''

import numpy as np
import matplotlib.pyplot as plt


# class Lagrange:
#     def __init__(self, x, fx):
#         self.__x = x
#         self.__fx = fx
#         self.__n = len(x)
#
#     def init(self):
#         l = []
#         for i in range(self.__n):
#             l_i =

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


from scipy.interpolate import lagrange

if __name__ == '__main__':
    x = [.0, .5, 1., 6., 7., 9.]
    fx = [0., 1.6, 2., 1.5, 1.5, 0.]
    lagrange = Lagrange(x, fx)

    # x = [.0, .5, 1., 6., 7., 9.]
    # y = [0., 1.6, 2., 1.5, 1.5, 0.]
    # lagrange(x, y)
    # for i in range(10):
    #     print(lag(i))
    #     print(lagrange(x, y)(i))

    t = [0.01 * i for i in range(1001)]
    lagrange_x = [lagrange(i) for i in t]

    plt.plot(x, fx, color='m', linestyle='', marker='o', label='给定数据')
    plt.plot(t, lagrange_x, color='b', linestyle='-', marker='', label="拉格朗日插值曲线")

    plt.show()