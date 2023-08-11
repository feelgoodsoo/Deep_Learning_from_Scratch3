
## 패키지로 정리 ##
# dezero 폴더 생성 후 core_simple.py 작성


if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero import Variable
import numpy as np


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
