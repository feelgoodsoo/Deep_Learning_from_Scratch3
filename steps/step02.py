import numpy as np

## 함수 구현 ##
## Function, Square 구현 ##


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward 메서드에서 한다
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


'''
## Example Test Codes ##
x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))
print(y.data)
'''
