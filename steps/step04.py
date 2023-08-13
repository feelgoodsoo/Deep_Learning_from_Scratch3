import numpy as np

## 수치 미분 구현 ##

# 기울기란 -> f(x+h)-f(x)/h
# 여기서 h를 0에 아주 가깝게 만들면 미분이 됨
# 0에 아주 가까운 값 h를 편의상 1e-4로 대체함 (수치 미분은 진정한 미분을 근사함)
# 수치 미분은 약간의 오차가 발생할 수 있기에 중앙차분을 이용함 -> f(x+h)-f(x-h)/2h


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


class Exp(Function):
    def forward(self, x):
        # e는 자연로그의 밑으로 구체적인 값은 2.718... (오일러 상수 혹은 네이피어 상수라고 불림)
        return np.exp(x)

## 수치 미분 ##


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


## test example codes ##
'''
f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)
'''

## 합성 함수 ##


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


## text example codes ##
x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
