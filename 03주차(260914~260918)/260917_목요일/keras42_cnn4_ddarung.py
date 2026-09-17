from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
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
scaler = MinMaxScaler()
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

print(x_train.shape, x_test.shape)  #(1062, 9) (266, 9)
print(y_train.shape, y_test.shape) #(1062,) (266,)
x_train = x_train.reshape(-1, 3, 3, 1)
x_test = x_test.reshape(-1, 3, 3, 1)
print(x_train.shape, x_test.shape)  #(1062, 3, 3, 1)
print(y_train.shape, y_test.shape)  #(266, 3, 3, 1)


#2-1 순차적 모델
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(3, 3, 1), padding='same'))  
model.add(Conv2D(32, (2,1), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,1), activation='relu', padding='same')) 
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))  
model.summary()

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
loss =  2315.137451171875
걸린시간 =  10.05 초
"""