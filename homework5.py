'''
P224-2
'''
import numpy as np
import matplotlib.pyplot as plt


def fit(x, y, n):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    A = np.array([[np.sum((x ** i).dot(x ** j)) for j in range(n + 1)] for i in range(n + 1)])
    b = np.array([np.sum(y.dot(x ** i)) for i in range(n + 1)])
    a = np.linalg.inv(A).dot(b)

    def func(new_x):
        return np.sum([a[i] * new_x ** i for i in range(0, n + 1)])

    return func


if __name__ == '__main__':
    x = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    y = [1.0, 0.41, 0.50, 0.61, 0.91, 2.02, 2.46]

    t = [0.001 * i for i in range(1001)]
    plt.scatter(x, y, marker='o', c='g', zorder=10)
    plt.plot(x, y, c='g', label='origin data')
    curve = fit(x, y, 3)
    plt.plot(t, [curve(i) for i in t], c='b', label='n=3')
    curve = fit(x, y, 4)
    plt.plot(t, [curve(i) for i in t], c='r', label='n=4')
    curve = fit(x, y, 5)
    plt.plot(t, [curve(i) for i in t], c='brown', label='n=5')
    curve = fit(x, y, 6)
    plt.plot(t, [curve(i) for i in t], c='black', label='n=6')
    plt.legend(loc='best')
    plt.show()
