'''
P116-1
'''

# x*
X_STAR = 1.368808107
# 误差
ERROR = 5e-10
# 允许的最大迭代次数
MAX_ITER = 1e4


# （1）中的迭代函数
def phi_1(x):
    return 20 / (x * x + 2 * x + 10)


# （2）中的迭代函数
def phi_2(x):
    return (20 - 2 * x * x - x ** 3) / 10


# 原方程
def f(x):
    return x ** 3 + 2 * x * x + 10 * x - 20


# 原方程的一阶导数
def f_deri(x):
    return 3 * x * x + 4 * x + 10


# 第一问
def question_1(x):
    cnt = 0
    # 当超过最大迭代次数时停止迭代，下同
    while cnt <= MAX_ITER:
        x_n = phi_1(x)
        cnt += 1
        if x_n == x:
            break
        x = x_n

    return x, cnt


# 第二问
def question_2(x):
    cnt = 0
    while cnt <= MAX_ITER:
        x_n = phi_2(x)
        cnt += 1
        if x_n == x:
            break
        x = x_n

    return x, cnt


# 第三问
def question_3(x):
    cnt = 0
    while cnt <= MAX_ITER:
        y = phi_1(x)
        # 当x和y的差值足够小时提前停止迭代
        if abs(x - y) < ERROR:
            break
        z = phi_1(y)
        x = x - (y - x) ** 2 / (z - 2 * y + x)
        cnt += 1

    return x, cnt


# 第四问
def question_4(x):
    cnt = 0
    while cnt <= MAX_ITER:
        y = phi_2(x)
        if abs(x - y) < ERROR:
            break
        z = phi_2(y)
        x = x - (y - x) ** 2 / (z - 2 * y + x)
        cnt += 1

    return x, cnt


# 第五问
def question_5(x):
    cnt = 0
    while cnt <= MAX_ITER:
        x_n = x - f(x) / f_deri(x)
        cnt += 1
        if x_n == x:
            break
        x = x_n

    return x, cnt


if __name__ == '__main__':
    x_0 = 1.0
    for i in range(1, 6):
        print('The result of question {}:'.format(i))
        print('Result: {}\nIter times: {}\n'.format(*globals()['question_{}'.format(i)](x_0)))
