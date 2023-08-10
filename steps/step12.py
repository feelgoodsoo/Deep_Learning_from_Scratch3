import numpy as np

## 가변 길이 인수(개선편) ##
## Function class 수정 및 Add class 수정 ##


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
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # *로 언팩
        if not isinstance(ys, tuple):  # 튜플이 아닌 경우 추가 지원
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        # 리스트의 원소가 하나라면 첫 번째 원소를 반환한다
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def add(x0, x1):
    return Add()(x0, x1)


## example test codes ##
'''
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1) # Add 클래스 생성 과정 추상화
print(y.data)
'''
