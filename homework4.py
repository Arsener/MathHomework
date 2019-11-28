'''
P190-2
'''

import matplotlib.pyplot as plt


# 拉格朗日插值
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

    # 返回函数表达式
    return polynomial


# 三次样条插值，二型边界条件，默认自然边界条件
def spline_II_edge(x, fx, M_0=0., M_n=0.):
    n = len(x)
    # h
    h = [x[i] - x[i - 1] for i in range(1, n)]
    # mu
    mu = [h[i - 1] / (h[i - 1] + h[i]) for i in range(1, n - 1)]
    # lambda
    lmbd = [1 - m for m in mu]

    # 1阶均差
    f_1_step = [(fx[i] - fx[i - 1]) / (x[i] - x[i - 1]) for i in range(1, n)]
    # 2阶均差 * 6
    f_2_step = [6 * (f_1_step[i - 1] - f_1_step[i - 2]) / (x[i] - x[i - 2]) for i in range(2, n)]
    # 方程式右边的向量
    f_2_step[0] -= mu[0] * M_0
    f_2_step[-1] -= lmbd[-1] * M_n

    # 追赶法求解三对角矩阵：AM = f_2_step
    # A = LU
    u = [2]
    l = []
    for i in range(1, n - 2):
        l.append(mu[i] / u[-1])
        u.append(2 - l[-1] * lmbd[i - 1])

    # Ly = f_2_step
    y = [f_2_step[0]]
    for i in range(1, n - 2):
        y.append(f_2_step[i] - l[i - 1] * y[-1])

    # UM = y
    M = [y[-1] / u[-1]]
    for i in range(n - 4, -1, -1):
        M = [1 / u[i] * (y[i] - lmbd[i] * M[0])] + M
    M = [M_0] + M + [M_n]

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

    # 返回函数表达式
    return polynomial


if __name__ == '__main__':
    x = [.0, .5, 1., 6., 7., 9.]
    fx = [0., 1.6, 2., 1.5, 1.5, 0.]
    spline = spline_II_edge(x, fx)
    lagrange = Lagrange(x, fx)

    t = [0.01 * i for i in range(901)]

    spline_x = [spline(i) for i in t]
    lagrange_x = [lagrange(i) for i in t]

    plt.scatter(x, fx, color='g', marker='o')
    plt.plot(t, lagrange_x, color='b', linestyle='-', label="Lagrange")
    plt.plot(t, spline_x, color='r', linestyle='-', label="Spline")

    plt.legend(loc='best')

    plt.show()
