if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import test_mode
import dezero.functions as F

## 드롭아웃과 테스트 모드 ##

# 드롭아웃은 과적합을 방지하기 위한 수단이다 -> 뉴런을 임의로 삭제(비활성화)하면서 학습 진행 #

# dezero/core.py Config 추가 구현
# dezero/functions.py 드롭아웃 구현

x = np.ones(5)
print(x)

# 학습시
y = F.dropout(x)
print(y)

# 테스트 시
with test_mode():
    y = F.dropout(x)
    print(y)
