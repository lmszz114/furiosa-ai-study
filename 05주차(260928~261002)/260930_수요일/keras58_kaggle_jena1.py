# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# wd 를 y로 -> T (degC) 로 변경(26-09-30 실습)
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

# 기존 wd
# x = bbb[:-144, :, :-1]
#y = bbb[:-144, -1, -1]

# 변경 (T = 1번 컬럼, datasets에서 index_col=0 으로 인덱스 제외시켰음)
x = np.delete(bbb[:-144], 1, axis=2)   # T(1번) 컬럼 제외 → (420264, 144, 13)
y = bbb[:-144, -1, 1]                   # 마지막 timestep의 T

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

# 기존 (wd)
# 예측 구간 = bbb의 마지막 144개 window (= 2016-12-31 00:10 ~ 2017-01-01 00:00)
# x_predict = bbb[-144:, :, :-1]      # (144, 144, 13)  wd 컬럼 제외
# y_true    = bbb[-144:, -1, -1]      # (144,)          실제 wd (정답 비교용)

# 변경 (T)
x_predict = np.delete(bbb[-144:], 1, axis=2)   # (144, 144, 13)  T 제외
y_true    = bbb[-144:, -1, 1]                   # 실제 T

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
T 예측
loss =  0.006699126213788986
걸린시간 =  3347.5 초
5/5 [==============================] - 0s 5ms/step
R2 :  0.9987376918285932
RMSE :  0.11727316499940836
2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 의 예측값(144개): 
 [-4.2384963  -4.231602   -4.1745358  -4.080996   -4.031606   -4.0763507
 -4.2816544  -4.5534515  -4.678091   -4.8919153  -5.0885983  -5.210777
 -5.2208247  -5.2525277  -5.334437   -5.2922945  -5.224013   -5.107781
 -4.794791   -4.2338133  -3.9829056  -3.8840723  -4.019743   -4.1175623
 -4.381083   -4.5765343  -4.767141   -4.941272   -5.0803328  -5.1445036
 -5.04405    -5.080505   -5.1676717  -5.107678   -5.0768285  -5.2094707
 -5.088773   -5.2093806  -5.47174    -5.83021    -5.9256926  -5.9863267
 -6.232209   -6.060253   -6.1602917  -6.432521   -6.488065   -6.756015
 -6.9013243  -6.761254   -6.827411   -6.485017   -6.3714457  -6.360147
 -6.34022    -6.344623   -6.1717286  -6.0203514  -5.7598133  -5.4116807
 -5.21613    -4.733789   -4.682314   -4.5858607  -4.2984877  -3.9238677
 -3.4355345  -3.102693   -2.6607337  -2.1016047  -1.8807179  -1.4165187
 -0.8772226  -0.41109407  0.31320304  1.044699    1.5619893   1.6290529
  1.8626419   2.0918474   2.4629812   2.9694622   3.7815661   4.045532
  4.2036777   4.412476    4.7319164   5.112709    5.2465982   5.165046
  5.018204    4.7410817   4.4820666   4.072964    3.6663077   3.020282
  2.4932146   2.1252928   1.8950902   1.580697    1.488465    1.3210249
  1.2236285   0.8058806   0.42149916  0.1595107  -0.06433725 -0.15966134
 -0.44597432 -0.8672747  -1.2916659  -1.6384258  -1.3699245  -1.1830232
 -1.1390479  -1.3082985  -1.4777998  -1.6210024  -1.6550242  -1.5147289
 -1.494213   -2.0184748  -2.9817822  -3.410489   -3.5139596  -3.2327676
 -2.9276261  -2.7185376  -2.613216   -2.570295   -2.567137   -2.6421194
 -2.874911   -3.103881   -3.8378482  -4.57844    -4.377421   -3.8183196
 -3.8281972  -4.022169   -3.54875    -3.2335496  -3.9563491  -4.7474675 ]
"""