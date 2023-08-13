if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

## 선형 회귀 ##

# 선형 회귀에서 출력 오차 값은 평균 제곱 오차로 구한다
# 평균 제곱 오차(Mean-squared-error) 값은 다음과 같다

# 1. 총 N개의 점에 대해 (xi, yi)의 각 점에서 제곱 오차를 모두 구한다
# 2. 오차들을 모두 더한다
# 3. 평균을 구하기 위해 N으로 나눈다

# 우리의 목표는 평균 제곱 오차가 최소가 되는 모델을 만드는 것이다
# 평균 제곱 오차가 최소가 되기 위해선 경사하강법으로 가중치(W), 편향(b)을 조정하여야 한다

# dezero/functions.py MeanSquaredError class 작성 #

# 토이 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss, W.grad.data, b.grad.data)


# Plot
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.show()
