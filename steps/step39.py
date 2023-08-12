## 합계 함수 ##

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

## dezero/functions.py sum 함수 구현 ##
# dezero/functions.py broadcast 함수 추가.. 원래는 step40에서 다룰 내용인데 땡겨쓴다고 함.. 후반부로 갈수록 구현 과정 순서가 짬뽕이 되어가는구나..
# dezero/utils.py reshape_sum_backward 추가.. ( numpy의 문제에 대한 대응이기에 책에는 구현 과정을 생략한다고 되어있음.. )
# broadcast 구현하며 sum_to도 먼저 구현

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.sum(x, axis=0)
print(y)
print(x.shape, '->', y.shape)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)
