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
# model.add(SimpleRNN(units=10, input_shape=(3,1)))
model.add(SimpleRNN(10, input_shape=(3,1)))
# 3차원으로 들어가서 1(or2)차원으로 나옴(None, 10) -> 바로 Dense와 연결가능
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss: ', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)
print('[8,9,10]의 결과: ', y_predict)

"""
목표: 예측값 11 만들어보기 
loss:  0.0010473739821463823
1/1 [==============================] - 0s 73ms/step
[8,9,10]의 결과:  [[10.756406]]
"""