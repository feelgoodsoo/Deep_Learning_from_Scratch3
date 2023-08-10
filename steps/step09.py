import numpy as np

## 함수를 더 편리하게 ##

# square func 생성
# exp func 생성
# Variable backward 메서드 간소화


class Variable:
    def __init__(self, data):
        if data is not None:  # ndarray만 취급하기
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.', format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 함수를 가져온다
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다
            x.grad = f.backward(y.grad)  # backward 메서드를 호출한다

            if x.creator is not None:
                funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward 메서드에서 한다
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input  # 입력 변수를 기억(보관)한다
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        # e는 자연로그의 밑으로 구체적인 값은 2.718... (오일러 상수 혹은 네이피어 상수라고 불림)
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


## example test codes ##
'''
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

x = Variable(np.array(0.5))
y. square(exp(square(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)
'''

'''
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)
'''

'''
x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
x = Variable(1.0)  # NG


x = np.array([1.0])
y = x ** 2
print(type(x), x.ndim)
print(type(y))

x = np.array(1.0)  # 0차원 ndarray
y = x**2
print(type(x), x.ndim)  # <<class 'numpy.ndarray'> 0
print(type(y))  # <<class 'numpy.float64' > ==> x**2하면 float64가 되버림
'''
