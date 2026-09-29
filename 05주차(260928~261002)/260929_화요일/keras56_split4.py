import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

a = np.array(range(1, 101))


size = 11

# [실습]
# 데이터를 reshape한 후, split_x 함수로 시계열데이터로 변환
# (N, 10, 1) -> (N, 5, 2)
# 결과 뽑기


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    # (90, 11)

x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape)  # (90, 10)
print(y.shape)  # (90,)

# print(x)
# print(y)

x = x.reshape(-1, 5, 2)
print(x.shape)  # (90, 5, 2)


#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(5,2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=4)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

x_predict = np.array(range(96,106))     
x_predict = x_predict.reshape(1, 5, 2) 
y_predict = model.predict(x_predict)

print('106 예측값: ', y_predict)

"""
loss:  0.009929344058036804
1/1 [==============================] - 0s 254ms/step
106 예측값:  [[104.281685]]
"""