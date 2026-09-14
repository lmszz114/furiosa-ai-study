# keras30_ModelCheckPoint1 카피

# import ssl
# ssl._create_default_https_context = ssl._create_default_https_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time

path = './_save/keras30/'


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=2048,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler.fit(x_train)
scaler = RobustScaler()
# x_train = scaler.transform(x_train) # 아래처럼 줄여서 한줄에 쓸 수도 있음
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(40, activation='relu', input_dim=8))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1,)

####################### mcp 세이브 파일명 만들기 시작 #######################
import datetime
date = datetime.datetime.now()
print(date) # 2026-09-14 11:40:56.443202
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) # 0914_1147

path = './_save/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path, "k30_", date, "-", filename])
# 저장되는 방식 예시: './_save/keras30/' + "k30_" + "0914_1147" + 530-0.0001.keras
####################### mcp 세이브 파일명 만들기 끝 #######################


mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath, verbose=1,)

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, verbose=1,
                 epochs=1000, batch_size=32, validation_split=0.2,
                 callbacks=[es, mcp])
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

print('loss = ', loss)
print("RMSE = ", rmse)

"""
RobustScaler
loss =  0.28122851252555847
RMSE =  0.530309790628071
"""