import numpy as np

## 역전파 자동화 ##
# Define-by-Run 방식 구현 #
# Define-by-Run이란 수행하는 계산들을 계산 시점에 '연결'하는 방식, '동적 계산 그래프'라고도 불림
# 'linked list' 데이터 구조를 이용하여 구현한다
# Variable set_creator 추가
# Function output 추가
# Variable backward 추가


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. 함수를 가져온다
        if f is not None:
            x = f.input  # 2. 함수의 입력을 가져온다
            x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드를 호출한다
            x.backward()  # 하나 앞 변수의 backward 메서드를 호출한다(재귀)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward 메서드에서 한다
        output = Variable(y)
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


## test Example Codes ##
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올라간다
'''
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x
'''

## 역전파 도전 ##
y.grad = np.array(1.0)

C = y.creator  # 1. 함수를 가져온다
b = C.input  # 2. 함수의 입력을 가져온다
b.grad = C.backward(y.grad)  # 3. 함수의 backward 메서드를 호출한다

B = b.creator  # 1. 함수를 가져온다
a = B.input  # 2. 함수의 입력을 가져온다
a.grad = B.backward(b.grad)  # 3. 함수의 backward 메서드를 호출한다

A = a.creator  # 1. 함수를 가져온다
x = A.input  # 2. 함수의 입력을 가져온다
x.grad = A.backward(a.grad)  # 3. 함수의 backward 메서드를 호출한다
print(x.grad)


## 역전파 자동화 테스트 ##
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)


# 역전파
y.grad = np.array(1.0)
y.backward()
print(x.grad)
