if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F

## CNN 메커니즘(1) ##

# CNN(Convolutional Neural Network) == 합성곱 신경망
# 구성 요소: 합성곱층(convolutional layer), 풀링층(pooling layer)

# conv -> reLU -> pooling -> conv -> reLU -> pooling -> conv ... 형태의 레이어가 반복
# 출력에 가까워지면 Linear -> reLU 조합이 사용됨

'''
합성곱층에서는 필터 연산이 진행된다.  쉽게 말하면 기존의 n*n 입력 데이터를 압축하여 새로운 n*n 데이터로 출력한다.
이 과정은 filter window를 일정 간격(stride)으로 이동시키며 진행된다
필터를 kernel이라고도 부른다.

합성곱층의 주요 처리 전에 입력 데이터 주위에 고정값(가령 0 등)을 채울 수 있는데 이 과정을 '패딩(padding)'이라고 한다.
'stride'는 필터를 적용하는 위치 간격을 의마한다
'''

# dezero/utils.py 에 아래 함수 추가


def get_conv_outsize(input_size, kernal_size, stride, pad):
    return (input_size + pad * 2 - kernal_size) // stride + 1  # // -> 몫 연산자


H, W = 4, 4  # 입력 형상
KH, KW = 3, 3  # 커널 형상
SH, SW = 1, 1  # 스트라이드(세로 방향 스트라이드와 가로 방향 스트라이드)
PH, PW = 1, 1  # 패딩(세로 방향 패딩과 가로 방향 패딩)

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)
