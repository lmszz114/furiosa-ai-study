import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer # 유방암 관련 데이터

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)   # 내역명세
# Number of Instances: 569
# Number of Attributes: 30 numeric, predictive attributes and the class
# 569행 30열 데이터
print(datasets.feature_names)   #피처의 이름들을 모두 출력

# x = datasets.data
x = datasets["data"]    # 딕셔너리 데이터이므로 이렇게도 불러올 수 있음
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>

#### 0이 몇개고, 1이 몇개인지 -> numpy
print(y) # [0 0 0 0 0 0 ... 1 1 1 0 0 0 ...] 몇개 안돼서 분류가 어떻게 되는지 식별 가능
# 데이터가 많으면 눈으로 식별하기 어려움 (0개 몇개고, 1이 몇개인지)
# 그래서 ↓
print(np.unique(y)) # [0 1] 분류의 종류가 어떤것들이 있는지만 출력 (중복되는 값이 여러번 안뜸)
# 하지만 각 분류(범주)들이 각각 몇개씩인지는 모름
# 그래서 ↓
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357])) -> 0은 212개, 1은 357개

#### 0이 몇개고, 1이 몇개인지 -> pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212
#### numpy, pandas 편한거 쓰면 됨 / 0 과 1이 몇개인지만 파악하면 되니까

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=32,
    stratify=y,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True))    # (array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)


#2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=30))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='binary_crossentropy', optimizer='adam',
              # metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1,)

####################### mcp 세이브 파일명 만들기 시작 #######################
import datetime
date = datetime.datetime.now()
print(date) # 2026-09-14 11:40:56.443202
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) # 0914_1147

path = './_save/keras31/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path, "k39_", date, "-", filename])

####################### mcp 세이브 파일명 만들기 끝 #######################


mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath, verbose=1,)

start_time = time.time()
hist = model.fit(x_train, y_train, verbose=1,
                 epochs=1000, batch_size=32, validation_split=0.2,
                 callbacks=[es, mcp])
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

"""
loss =  0.026433715596795082
acc =  0.9708
"""
