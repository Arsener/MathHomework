from sympy import *

x = Symbol('x')
d = diff((x ** 3 - x) ** 2, x, 4)
print(d)

ans = solve(d, x)
print(ans)

for a in ans:
    print(a.evalf())