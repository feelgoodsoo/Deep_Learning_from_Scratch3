import numpy as np
import unittest

## 테스트 케이스 구현 ##


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

## 수치 미분 ##


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

## 합성 함수 ##


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

# test by using 'python3 -m unittest steps/step10.py'
