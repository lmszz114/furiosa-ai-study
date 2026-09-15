from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
path = "./_data/ddarung/"  # 상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0) # index_col=0 첫번째 컬럼을 인덱스로 씀
print(train_csv) 
# 열 미포함 [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #test.csv 에는 count 컬럼이 없음 -> y값이 없다는 말임
print(test_csv)
# [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)
# [715 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())

train_csv = train_csv.dropna() # [1328 rows x 10 columns]
print(train_csv) 

# ★중요★ train_csv를 x와 y로 분리
# train_csv 에서 count 컬럼을 제거 -> coount 가 y 값이 되는거임 (이 문제에서 요구하는 예측치값)

x = train_csv.drop(['count'], axis=1) # axis: 축 / 열삭제(컬럼) / count 컬럼을 제외한 나머지는 x에 입력
print(x) # (1328, 9)

y = train_csv['count'] # x에서 빼놨던 count 를 y에 입력
print(y)
print(y.shape) # (1328,) 벡터 형태

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=484,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

################ submit 사전작업 ################ 
print(test_csv.info()) # 컴파일 시 결측치 있는걸 확인할 수 있음

############### 결측치(non-null) 처리 2. 평균값 넣기 ###############
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info()) 
print(test_csv.shape)   #(715, 9)

#2-1 순차적 모델
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=9))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
model.summary()
###########################################################
"""
#2-2 함수형 모델
input1 = Input(shape=(9,))
dense1 = Dense(64, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(32, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(16, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(8, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(4, activation='relu')(drop4)
dense6 = Dense(2, activation='relu')(dense5)
output1 = Dense(1)(dense6)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()
"""

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 999,    
    restore_best_weights = True,
)

import time

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=100, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")


"""
CPU
loss =  3133.56201171875
걸린시간 =  8.65 초

GPU
loss =  18615.333984375
걸린시간 =  7.64 초

GPU가 빠름
"""