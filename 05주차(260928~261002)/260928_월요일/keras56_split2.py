import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]
              ]).T
print(a.shape)  # (10, 2)

# 스플릿으로 잘라보기

size = 5    # timestep 사이즈

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    # (6, 5, 2)

x = bbb[:, :-1, :]
# x = bbb[:, :-1]   # 이렇게도 사용 가능, 가독성은 안좋음
y = bbb[:, -1, 1]   # 마지막은 피처니까 0으로 하면 0번째 피처, 1로 하면 1번째 피처 찾는거임
# y = bbb[:, -1, -1]   # 이렇게도 사용 가능

print("=========================")
print(x)
print("=========================")
print(y)    # [5 4 3 2 1 0]
print("=========================")
print(x.shape)  # (6, 4, 2)
print(y.shape)  # (6,)


#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(4,2)))
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
x_predict = np.array([[7,3],[8,2],[9,1],[10,0]]).reshape(1,4,2)
y_predict = model.predict(x_predict)
print('[7,3],[8,2],[9,1],[10,0]의 예측값: ', y_predict)

"""
[7,3],[8,2],[9,1],[10,0]의 예측값:  [[-0.19039735]]
"""