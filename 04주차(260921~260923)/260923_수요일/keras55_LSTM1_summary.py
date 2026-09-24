#54_2 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],  # 타임스텝스 3으로 자름
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

y = np.array([4,5,6,7,8,9,10])  # y 데이터를 직접 만듬 (왜? y데이터가 없음)
print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)    # x데이터를 3차원으로 reshape
print(x.shape)  # (7, 3, 1)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(5, input_shape=(3,1)))
model.add(LSTM(10, input_shape=(3,1)))
# RNN 계열중에 LSTM이 가장 강력함
# RNN 의 문제점: 앞을 까먹음 (데이터가 많아질수록 불리해짐) -> 이걸 개선하기 위해 LSTM 사용 
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.summary()

"""
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10)                480       
                                                                 
 dense (Dense)               (None, 7)                 77        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0
_________________________________________________________________
"""