# 53_06 카피

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer # 유방암 관련 데이터

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)   # 내역명세
print(datasets.feature_names)   #피처의 이름들을 모두 출력

# x = datasets.data
x = datasets["data"]    # 딕셔너리 데이터이므로 이렇게도 불러올 수 있음
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>

print(y) 
print(np.unique(y)) 
print(np.unique(y, return_counts=True)) 

#### 0이 몇개고, 1이 몇개인지 -> pandas
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=32,
    stratify=y, # 분류된 데이터에서 비율에 맞춰서 라벨링 해준다 (라벨에 따라서 데이터가 골고루 분포되어야함, train/test size에 맞춰서 나눠줌)
)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 30, 1)
x_test  = x_test.reshape(-1, 30, 1)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(np.unique(y_train, return_counts=True))  
print(np.unique(y_test, return_counts=True)) 

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(LSTM(40, input_shape=(30, 1)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()
###########################################################


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01


model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5  # 0.5=반띵
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 999,    
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=100, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es,rlr],
                 )
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print("===========================================")
print('loss = ', loss[0])
print('acc = ', round(loss[1],4))
print("===========================================")



y_pred = model.predict(x_test)
y_pred = np.round(y_pred)   # round 처리 해줘야 안터짐

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score = ", acc_score)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
ReduceLROnPlateau 적용
learning_rate = 0.01
acc_score =  0.9590643274853801
걸린시간 =  4.3 초
"""

"""
LSTM으로 재구성
acc_score =  0.9415204678362573
걸린시간 =  8.16 초
"""