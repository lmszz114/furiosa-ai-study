from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D


model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))    # (2,2) = 필터 사이즈 / 행의 크기 제외하고 input_shape 로 잡는다
model.add(Conv2D(5, (2,2)))

model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50        
                                                                 
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       
                                                                 
=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0

5,5 사이즈 데이터가 2,2 커널을 통과하면 4,4가 된다.
"""