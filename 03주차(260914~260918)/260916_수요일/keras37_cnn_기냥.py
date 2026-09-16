from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D


model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))    # (2,2) = 필터 사이즈 / 행의 크기 제외하고 input_shape 로 잡는다
model.add(Conv2D(5, (2,2)))

model.summary()

