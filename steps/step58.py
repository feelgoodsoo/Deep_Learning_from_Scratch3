if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
from dezero.core import Variable
from dezero.models import VGG16
from PIL import Image
from dezero import utils
import dezero

## VGG16 구현 ##

# dezero/models.py VGG16 구현
# dezero/functions_conv.py pooling 추가
# dezero/datasets.py ImageNet 추가

model = VGG16(pretrained=True)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)  # 더미 데이터
model.plot(x)  # 계산 그래프 시각화

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'

img_path = utils.get_file(url)
img = Image.open(img_path)
img.show()

x = VGG16.preprocess(img)
print(type(x), x.shape)

x = x[np.newaxis]  # 배치용 축 추가

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels = dezero.datasets.ImageNet.labels()  # 이미지넷의 레이블
print(labels[predict_id])
