## 변수 Variable 구현 ##

class Variable:
    def __init__(self, data):
        self.data = data


## example Test Codes ##
'''
import numpy as np

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)
'''
