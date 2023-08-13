if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP

## 소프트맥스 함수와 교차 엔트로피 오차 ##

# 다중 클래스 분류 도전 #

# 다중 클래스 분류에서 사용되는 손실 함수는 교차 엔트로피 오차(cross-entroy-error)이다
# 교차 엔트로피 오차를 표현하기 위해 원핫 벡터(one-hot-vector) 방식을 이용한다

# dezero/functions.py get_item 구현
# -> core.py의 setupVariable 추가 (__get__item)
# dezero/functions.py softmax_simple, softmax_corss_entropy_simple 구현
# dezero/functions.py log, clip 구현

np.random.seed(0)

model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
print(y)


def sofmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = sofmax1d(y)
print(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])

y = model(x)
p = F.softmax_simple(y)
print(y)
print(p)

loss = F.softmax_cross_entropy_simple(y, t)
loss.backward()
print("loss: ", loss)
