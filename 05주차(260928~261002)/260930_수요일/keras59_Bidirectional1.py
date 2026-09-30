# 54_1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, Bidirectional, LSTM

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
model.add(Bidirectional(LSTM(64), input_shape=(3,1)))  #SimpleRNN 을 양쪽(양방향)으로 작업 / 괄호 주의
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary() # 양방향인지 확인

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500, batch_size=1)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)
print('[8,9,10]의 결과: ', y_predict)

"""
Bidirectional 적용

loss:  0.003157219151034951
1/1 [==============================] - 0s 295ms/step
[8,9,10]의 결과:  [[10.977501]]
"""