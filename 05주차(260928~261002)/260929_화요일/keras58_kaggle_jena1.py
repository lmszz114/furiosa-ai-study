# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# wd 를 y로
# 2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 기간의 wd 맞추기 (144개, 타임스텝스=144)
# 데이터에 포함되어있는 2016년12월31일 데이터는 모두 자르고 할것 (과적합 방지) (144개 자르면 됨, 완전 Drop)

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, SimpleRNN, GRU
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

x = bbb[:-144, :, :-1]
y = bbb[:-144, -1, -1]

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
model.add(LSTM(256, input_shape=(144,13)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
#learning_rate = 0.01
learning_rate = 0.001     # 0.001 - 디폴트
# learning_rate = 0.0005
#learning_rate = 0.005
#learning_rate = 0.05

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
                 epochs=100, 
                 batch_size=5000, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

# 예측 구간 = bbb의 마지막 144개 window (= 2016-12-31 00:10 ~ 2017-01-01 00:00)
x_predict = bbb[-144:, :, :-1]      # (144, 144, 13)  wd 컬럼 제외
y_true    = bbb[-144:, -1, -1]      # (144,)          실제 wd (정답 비교용)

# 학습 때 쓴 scaler로 transform만 (3D → 2D → 3D, x_train 때랑 동일)
x_predict = x_predict.reshape(-1, f)
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(-1, t, f)   # (144, 144, 13)

y_predict = model.predict(x_predict)      # (144, 1)

r2   = r2_score(y_true, y_predict)
rmse = np.sqrt(mean_squared_error(y_true, y_predict))
print('R2 : ', r2)
print('RMSE : ', rmse)
print('2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 의 예측값(144개): \n', y_predict.reshape(-1))


"""
loss =  6658.44384765625
걸린시간 =  2730.51 초
5/5 [==============================] - 0s 2ms/step
R2 :  -0.12989494376711264
RMSE :  57.24565511021961
2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 의 예측값(144개): 
 [182.50314 182.44511 182.31416 182.14682 182.11018 182.17397 182.00275
 181.64684 181.36191 180.9348  180.40948 179.82532 179.23497 178.76044
 178.32715 177.76816 177.09537 176.47607 175.91544 175.45909 175.07637
 174.62094 174.20338 173.76448 173.38553 173.0592  172.78485 172.56242
 172.33145 172.28159 172.11765 171.71573 171.31879 171.0216  170.90396
 170.97697 171.15082 171.3332  171.3121  170.9488  170.7098  170.82674
 170.84094 170.9505  171.09335 170.98534 170.88359 170.88794 170.90837
 170.90402 170.95436 170.94604 170.97519 170.97519 170.94138 170.85657
 170.83241 170.76825 170.5975  170.44753 170.09808 169.71696 169.26924
 168.68904 168.12128 167.46198 166.69293 165.71461 164.61835 163.43979
 162.18518 160.66197 158.98271 157.13837 155.34404 153.89334 153.59283
 154.6178  156.6422  159.15517 162.002   165.2736  168.51912 171.43848
 173.61636 174.98431 175.57509 175.53476 175.43561 175.48698 175.2001
 174.38736 173.54704 173.58105 175.31418 178.0385  180.24046 181.74074
 182.89995 183.84581 185.0221  186.44658 188.0467  189.72444 191.14424
 191.84007 191.85684 191.10445 189.72153 188.27667 186.69933 185.08424
 183.86458 182.76587 181.70175 181.1137  180.7707  180.56036 180.58598
 180.95105 181.37639 181.79984 182.1966  182.41899 182.73361 183.2074
 183.55695 183.94801 184.28172 184.56342 184.78262 184.92206 185.00793
 185.17728 185.33594 185.11618 184.95955 185.0656  185.20369 185.15492
 184.97021 184.81165 184.60947 184.30202]
"""
