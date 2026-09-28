import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(x.shape[0], x.shape[1], 1)    # (13, 3, 1)
print(x.shape)


#2. 모델구성
model = Sequential()
#model.add(LSTM(128, input_shape=(3,1)))
model.add(GRU(128, input_shape=(3,1)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss: ', result)

x_predict = np.array([50,60,70]).reshape(1,3,1)    # 80 맞추기
y_predict = model.predict(x_predict)
print('[50,60,70]의 결과: ', y_predict)

"""
[50,60,70]의 결과:  [[79.85066]]
"""