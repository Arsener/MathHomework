'''
P275-2
'''
import numpy as np
from sympy import *


def func(x):
    return (10. / x) ** 2 * np.sin(10. / x)


def Gauss_Legendre(f, a, b, n, h=1):
    """

    :param f: 需要计算积分的函数
    :param a: 积分区间左端
    :param b: 积分区间右端
    :param n: 代数精度
    :param h: 将区间分为几份，默认为1
    :return sum: 积分结果
    """
    n += 1
    # 计算f(x)分别取1,x,x^2,...时在-1到1上的积分结果
    fx = np.array([2 / (i + 1) * ((i + 1) % 2) for i in range(n)], dtype=np.float64)
    # 计算P_n(x)的零点
    x = Symbol('x')
    P = (x ** 2 - 1) ** n
    zero_point = np.array(sorted([s.evalf() for s in solve(diff(P, x, n), x)]), dtype=np.float64)
    # 计算A_0到A_n
    A = np.array([zero_point ** i for i in range(n)], dtype=np.float64)
    A = np.linalg.inv(A).dot(fx)

    # 分区间，并在每个区间上分别积分再求和
    interval = [a + i * (b - a) / h for i in range(h + 1)]
    sum = 0
    for sub_a, sub_b in zip(interval[0:-1], interval[1:]):
        sum += (sub_b - sub_a) / 2 * np.sum(A.dot(f(zero_point
                                                    * (sub_b - sub_a) / 2
                                                    + (sub_a + sub_b) / 2)))

    return sum


if __name__ == '__main__':
    ans = Gauss_Legendre(func, 1, 3, 4, 10)
    print('The integral result of the function on 1 to 3 is{}'.format(ans))

