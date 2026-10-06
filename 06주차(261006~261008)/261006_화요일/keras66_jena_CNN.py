# [실습] jena 데이터를 CNN 으로 재구성하기
# 59_03 카피
# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016


import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, SimpleRNN, GRU, Bidirectional, Reshape, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  #메모리 모으기

#1. 데이터
path = "./_data/kaggle_jena/"

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(datasets, size)
print(bbb)
print(bbb.shape)    # (420408, 144, 14)

x = np.delete(bbb[:-144], 1, axis=2)  
y = bbb[:-144, -1, 1]

print(x.shape)  # (420264, 144, 13)
print(y.shape)  # (420264,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=42,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()

n_train, t, f = x_train.shape     # (N_train, 144, 13)
n_test = x_test.shape[0]

x_train = x_train.reshape(-1, f)  # (N_train*144, 13)
x_test  = x_test.reshape(-1, f)   # (N_test*144, 13)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)

x_train = x_train.reshape(n_train, t, f)
x_test  = x_test.reshape(n_test,  t, f)

print(np.min(x_train), np.max(x_train))  # 0.0 1.0000000000000004
print(np.min(x_test),  np.max(x_test))  # 0.0 1.0000000000000004


#2. 모델구성
model = Sequential()
model.add(Reshape(target_shape=(144, 13, 1), input_shape=(144, 13))) # Dense 없이 처음부터 Reshape
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.005

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 30,    
    restore_best_weights = True,
)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=50, 
                 batch_size=256, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

x_predict = np.delete(bbb[-144:], 1, axis=2)
y_true    = bbb[-144:, -1, 1] 

x_predict = x_predict.reshape(-1, f)
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(-1, t, f)

y_predict = model.predict(x_predict)

r2   = r2_score(y_true, y_predict)
rmse = np.sqrt(mean_squared_error(y_true, y_predict))
print('R2 : ', r2)
print('RMSE : ', rmse)
print('2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 의 예측값(144개): \n', y_predict.reshape(-1))

"""
T 예측
loss =  0.006699126213788986
걸린시간 =  3347.5 초
5/5 [==============================] - 0s 5ms/step
R2 :  0.9987376918285932
RMSE :  0.11727316499940836
"""
"""
CNN으로 재구성 후 결과
loss =  0.13978689908981323
걸린시간 =  623.58 초
5/5 [==============================] - 0s 13ms/step
R2 :  0.9817633618952265
RMSE :  0.4457469080133982
"""