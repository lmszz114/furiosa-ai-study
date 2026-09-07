from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np


#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 8개, 4개, 4개 잘라보기
x_train = x[:8]
y_train = x[:8]

x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]

print(x_train.shape, x_val.shape, x_test.shape) # (8,) (4,) (4,)