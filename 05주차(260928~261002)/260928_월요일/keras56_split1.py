import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

a = np.array(range(1, 11))  # 1부터 10까지 (10개)

size = 5    # timestep 사이즈
print(a.shape)  # (10,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)

print(bbb.shape)    # (6, 5)
# 타임스텝 자르는 함수 꼭 이해할 것

x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape)  # (6, 4)
print(y.shape)  # (6,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (6, 4, 1)


#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(4,1)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)


#4. 평가, 예측
result = model.evaluate(x, y)
print('loss: ', result)

x_predict = np.array([7,8,9,10]).reshape(1,4,1)
y_predict = model.predict(x_predict)
print('7,8,9,10의 예측값: ', y_predict)

"""
7,8,9,10의 예측값:  [[10.288128]]
"""