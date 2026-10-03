# 53_01 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(-1, 8, 1)
print(x.shape) # (20640, 8, 1)

print(np.min(x), np.max(x))
# 0.0 1.0000000000000002

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=2048,
)

print(x_train.shape, x_test.shape)  # (15480, 8) (5160, 8)
print(y_train.shape, y_test.shape)  # (15480,) (5160,)



#2. 모델구성
model = Sequential()
model.add(LSTM(40, input_shape=(8, 1)))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001     # 0.001 - 디폴트

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5  # 0.5=반띵
)

hist = model.fit(x_train, y_train, verbose=1, epochs=500, batch_size=32, validation_split=0.2, 
                 callbacks=[es,rlr])

print("===================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):    # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt << 루트 씌우기
rmse = RMSE(y_test, y_predict)

print("=============== loss ====================")
print(hist.history['loss'])
print("=============== val_loss ====================")
print(hist.history['val_loss'])
print("=============== loss, rmse ====================")
print('loss = ', loss)
print("RMSE = ", rmse)

"""
ReduceLROnPlateau 적용
learning_rate = 0.01
loss =  0.5392395257949829
RMSE =  0.7343292215851057
"""

"""
LSTM으로 재구성
loss =  0.3714119791984558
RMSE =  0.6094357841162215
"""