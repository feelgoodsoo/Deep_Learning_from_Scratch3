if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

## 매개변수를 모아두는 계층 ##

# 매개변수의 기울기를 재설정할 때 수동으로 코드를 작성해야 했음
# 매개변수 관리 자동화를 위해 Parameter, Layer class 생성

# dezero/core.py class Parameter 작성
# dezero/layers.py  Layer class 생성
# dezerp/layers.py  Linear 클래스 구현

# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

l1 = L.Linear(10)  # 출력 크기 지정
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print("loss: ", loss)
