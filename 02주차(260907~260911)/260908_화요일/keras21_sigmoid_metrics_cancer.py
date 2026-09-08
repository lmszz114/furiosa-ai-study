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
    stratify=y, # 분류된 데이터에서 비율에 맞춰서 라벨링 해준다 (라벨에 따라서 데이터가 골고루 분포되어야함, train/test size에 맞춰서 나눠줌)
)
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
model.compile(loss='binary_crossentropy', optimizer='adam',
              # metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 15,    
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=1, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
# loss =  0.18249453604221344
# 이 loss 는 잘 나온걸까? 알수없음
# loss는 상대적 판단 
# metrics=['accuracy'] 적용 후: loss =  [0.23196133971214294, 0.9005848169326782]
print("===========================================")
print('loss = ', loss[0])
print('acc = ', round(loss[1],4))
print("===========================================")


"""
y_pred = model.predict(x_test)
# print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score = ", acc_score)
# 에러처리됨 -> y_test는 0과1로 되어있음 / y_pred 는 0~1 사이 값으로 되어있음.
# ValueError: Classification metrics can't handle a mix of binary and continuous targets
"""
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)   # round 처리 해줘야 안터짐

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score = ", acc_score)