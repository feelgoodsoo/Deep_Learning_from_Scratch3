if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
from dezero.core import Variable

## conv2d 함수와 pooling 함수 ##
# dezero/functions_conv.py im2col, col2dim 모듈들, conv2d_simple 구현 ( conv2d도 추가 )
# dezero/functions.py 하단의 import 추가
# dezero/layers.py Conv2d 구현
# dezero/functions_conv.py pooling_simple 구현
# dezero/core.py transpose 수정
# dezero/functions.py transpose 수정

x1 = np.random.rand(1, 3, 7, 7)  # 배치 크기 = 1
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)  # 배치 크기 = 10
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)

N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)

x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()

print(x.shape)
print(x.grad.shape)
