# 36_2 카피

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

"""
######## 스케일링1 ########
x_train = x_train/255.  # 255. float
x_test = x_test/255.
# 이미지 데이터기 때문에 최대 범위가 255로 이미 고정되어있기 때문에 MinMax 안쓰고 255로 나눔
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0
# 범위: 0 ~ 1 사이가 됨 / MinMax 기능과 같음
"""

######## 스케일링2 ########
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0
# 범위: -1 ~ 1 사이가 됨 / MaxAbs 기능과 같음

# 통상적으로 이 두가지를 많이 씀
# 내일 이 코드 이어서 완성 예정