'''
P275-3
'''
import numpy as np


def func(x):
    return (10. / x) ** 2 * np.sin(10. / x)
    # return np.exp(x)


def Romberg(f, a, b, e=1e-4):
    h = b - a

    # 计算T(j,0)
    def t(j):
        n = 2 ** j
        sub_h = h / n / 2
        x = [a + i * h / n for i in range(int(n) + 1)]
        sum = 0
        for sub_a, sub_b in zip(x[0:-1], x[1:]):
            sum += sub_h * (f(sub_a) + f(sub_b))
        return sum

    T = [t(0)]
    j = 1
    T_tabel = 'T(0,k): ' + str(T)[1:-1] + '\n'
    while True:
        new_T = [t(j)]
        base = 4
        for i in range(len(T)):
            new_T.append((base * new_T[i] - T[i]) / (base - 1))
            base *= 4

        pre = T[-1]
        T = new_T
        T_tabel += 'T({},k): '.format(j) + str(T)[1:-1] + '\n'
        if abs(T[-1] - pre) < e:
            break

        j += 1

    return T[-1], T_tabel


if __name__ == '__main__':
    ans, T_tabel = Romberg(func, 1, 3)
    print('The T-tabel is:')
    print(T_tabel)
    print('The integral result of the function on 1 to 3 is {}'.format(ans))
