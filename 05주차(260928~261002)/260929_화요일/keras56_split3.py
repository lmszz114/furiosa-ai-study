import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

a = np.array(range(1, 101))
# [실습]
# 101 ~ 106까지 찾기 
# # loss 지표는 0.1 이하
# 결과는 [101, 102, 103, 104, 105, 106]의 근사치가 나오면 됨

size = 6

print(a.shape)  # (100,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    # (95, 6)

x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape)  # (95, 5)
print(y.shape)  # (95,)

# print(x)
# print(y)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (95, 5, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(5,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

x_predict = np.array(range(96, 106))       
x_predict = split_x(x_predict, 5)
x_predict = x_predict.reshape(-1, 5, 1) 

y_predict = model.predict(x_predict)
print('101~106 예측값: ', y_predict)

"""
loss:  0.000978653784841299
1/1 [==============================] - 0s 170ms/step
101~106 예측값:  [[100.81554 ]
 [101.68181 ]
 [102.4986  ]
 [103.24538 ]
 [103.91818 ]
 [104.560555]]
"""