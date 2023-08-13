if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F

## CNN 메커니즘(1) ##


def get_conv_outsize(input_size, kernal_size, stride, pad):
    return (input_size + pad * 2 - kernal_size) // stride + 1  # // -> 몫 연산자


H, W = 4, 4  # 입력 형상
KH, KW = 3, 3  # 커널 형상
SH, SW = 1, 1  # 스트라이드(세로 방향 스트라이드와 가로 방향 스트라이드)
PH, PW = 1, 1  # 패딩(세로 방향 패딩과 가로 방향 패딩)

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)
