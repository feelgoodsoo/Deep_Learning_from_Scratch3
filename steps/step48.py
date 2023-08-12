if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero
import dezero.functions as F
import math
from dezero import optimizers
from dezero.models import MLP
import matplotlib.pyplot as plt

## 다중 클래스 분류 ##

# dezero/datasets.py get_spiral 추가
# dezero/utils.py logsumexp, max_backward_shape 추가
# dezero/functions.py cross_entropy, softmax 추가
# dezero/functions.py max, min 추가

# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 스파이럴 데이터셋 로드 및 모델, 최적화 함수 생성 #
x, t = dezero.datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)  # 소수점 반올림

for epoch in range(max_epoch):
    # 데이터셋의 인덱스 뒤섞기
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 미니배치 생성
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        # softmax_cross_entropy_simple로 학습할 경우 loss 최적화가 안되기 때문에 softmax_cross_entropy 함수 사용
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        print("loss.data ", loss.data)
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # 에포크마다 학습 경과 출력
    print("sum_loss is", sum_loss, "len(batch_t) is ", len(batch_t))
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

# Plot boundary area the model predict
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()
