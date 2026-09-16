import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),  # 9, 9, 10
                 strides=2, # strides는 1이 디폴트
                 padding='same',))  # (None, 10, 10, 10)
model.add(Conv2D(filters=9, kernel_size=(3,3),  # 7, 7, 9
                 strides=2,
                 padding='valid',)) # padding은 valid가 디폴트 (valid: 패딩을 적용시키지 않는다) ->  (None, 8, 8, 9) 
model.summary()
# strides: 보폭
