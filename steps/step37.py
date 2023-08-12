## 텐서를 다루다 ##

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

# 텐서 사용시의 역전파 #
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
# y = sum(t)
# y.backward(retain_grad=True)
# print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)

# ** 아직 Sum class 작성을 다루지 않았기에 테스트는 불가..
