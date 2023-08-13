if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

## 신경망 ##

# 신경망의 기본 계산 -> y = F.matmul(x, W) + b
# 비선형 데이터셋의 문제를 선형 회귀로는 풀 수 없기 때문에 신경망을 도입하여야 한다
# 신경망은 선형 변환의 출력에 비선형 변환을 수행한다
# 비선형 변환을 활성화 함수(activation function)라고 부른다
# 대표적으로 reLU, sigmoid 등이 있다
# 일반적인 신경망은 선형 변환 -> 활성화 함수 -> 선형 변환 -> 활성화 함수가 반복되는 형태다

# dezero/functions.py linear_simple, sigmoid 함수 구현
# dezero/functions.py exp도 추가됨..


# 신경망 학습 코드 #

# 데이터셋 준비
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)  # 데이터 생성에 sin 함수 이용
print("y: ", y)

# 가중치 초기화
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# 신경망 추론


def predict(x):
    y = F.linear_simple(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear_simple(y, W2, b2)
    return y


lr = 0.2
iters = 10000


# 신경망 학습
for i in range(iters):
    y_pred = predict(x)
    # print("y_pred: ", y_pred)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:  # 1000회마다 출력
        print("loss: ", loss)

# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
