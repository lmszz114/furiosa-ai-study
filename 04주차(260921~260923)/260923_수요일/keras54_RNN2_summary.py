#54_1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

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
model.add(SimpleRNN(5, input_shape=(3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 5)                 35        
                                                                 
 dense (Dense)               (None, 7)                 42        
                                                                 
 dense_1 (Dense)             (None, 1)                 8         
                                                                 
=================================================================
Total params: 85
Trainable params: 85
Non-trainable params: 0
_________________________________________________________________
"""
'''
1. 파라미터 개수가 왜 이렇게 나오는지
파라미터의 개수: units*feature + units*bias + units*units
                  5*1        +    5*1     +     5*5

2. input_shape(3,1) 에서 3,1 은 무엇을 의미하는지
 3 -> timestep
 1 -> feature
'''