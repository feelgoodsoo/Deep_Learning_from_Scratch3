## 브로드 캐스트 함수 ##

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

# 원래 broadcast_to, sum_to를 현재 챕터에서 구현하기로 되어있었는데 step39에서 땡겨썻기에 이전 챕터에서 대부분 구현하고
# dezero/core.py의 Add class만 살짝 손볼 것이다..
# ** dezero/utils.py에 sum_to 함수 추가.. (np 대응용.. )

'''
** 책대로 진행하면 다음과 같은 에러가 발생함

Traceback (most recent call last):
  File "/Users/chlvlftn22/Desktop/AI/deepLearning_scratch/Deep_Learning_from_Scratch3/steps/step40.py", line 19, in <module>
    y.backward()
  File "/Users/chlvlftn22/Desktop/AI/deepLearning_scratch/Deep_Learning_from_Scratch3/steps/../dezero/core.py", line 99, in backward
    gxs = f.backward(*gys)
          ^^^^^^^^^^^^^^^^
  File "/Users/chlvlftn22/Desktop/AI/deepLearning_scratch/Deep_Learning_from_Scratch3/steps/../dezero/core.py", line 180, in backward
    gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
          ^^^^^^^^^^^^^^^^
AttributeError: module 'dezero' has no attribute 'functions'. Did you mean: 'Function'?

'''

'''
2시간동안의 샆질동안 찾아낸 해결법 :

dezero/__init__.py 의 else 이하 절에 다음의 코드 추가 
    
"from dezero import functions"
'''
x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)
