import numpy as np

## 가변 길이 인수(순전파편) ##
## Function class 수정 및 Add class 추가 ##


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
    def __call__(self, inputs):
        xs = [x.data for x in inputs]   # 입력 매개변수를 단일 인수에서 다중 인수로 변경
        ys = self.forward(xs)            # 순전파 계산 진행
        outputs = [Variable(as_array(y)) for y in ys]  # 위의 계산 결과를 Variable에 입력

        for output in outputs:
            output.set_creator(self)  # output의 set_creator 속성에 위의 결과들을 저장
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y, )  # 튜플 반환


## example test codes ##
xs = [Variable(np.array(2)), Variable(np.array(3))]  # 리스트로 준비
f = Add()
ys = f(xs)  # ys 튜플
y = ys[0]
print(y.data)
