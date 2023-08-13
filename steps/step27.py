
import math
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

## 테일러 급수 미분 ##

# 테일러 급수란 어떤 함수를 다항식으로 근사하는 방법 #
# 1차 미분 + 2차 미분.. + n차 미분까지의 값 근사 가능
# -> f(x) = f(a) + f'(a)(x-a) + 1/2! * f''(a)(x-a)^2 + 1/3! * f'''(a)(x-a)^3 ...
# 위의 수식에서 !는 factorial

# a=0일 때의 테일러 급수를 매클로린 전개(Maclaurin's series)라고 함

# sin 함수 구현 #

import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


## example test codes ##
x = Variable(np.array(np.pi/4))
y = sin(x)
y.backward()

print(y.data)
print(x.grad)

y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)

plot_dot_graph(y, verbose=False, to_file='my_sin.png')
