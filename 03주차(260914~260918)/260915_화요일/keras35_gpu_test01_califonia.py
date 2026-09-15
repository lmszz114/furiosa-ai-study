
import ssl
ssl._create_default_https_context = ssl._create_default_https_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2-1 순차적 모델
model = Sequential()
model.add(Dense(40, activation='relu', input_dim=8))
model.add(Dropout(0.2)) # 위에 있는 히든레이어(40)에 적용됨
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()
###########################################################
"""
#2-2 함수형 모델
input1 = Input(shape=(8,))
dense1 = Dense(40, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(40, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(30, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(20, activation='relu')(drop4)
drop5 = Dropout(0.1)(dense5)
dense6 = Dense(20, activation='relu')(drop5)
output1 = Dense(1)(dense6)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()
"""

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1,)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=path + 'keras30_mcp1.keras', verbose=1,)

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, verbose=1,
                 epochs=100, batch_size=32, validation_split=0.2,
                 callbacks=[es, mcp])
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

print('loss = ', loss)
print("RMSE = ", rmse)
print('걸린시간 = ', round(end_time - start_time, 2), "초")


"""
CPU
loss =  0.3213410973548889
RMSE =  0.5668696719707905
걸린시간 =  40.81 초

GPU
loss =  0.31525304913520813
RMSE =  0.5614739861627184
걸린시간 =  70.73 초

CPU가 빠름
"""
