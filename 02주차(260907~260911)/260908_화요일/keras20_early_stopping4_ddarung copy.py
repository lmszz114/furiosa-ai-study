from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

################ submit 사전작업 ################ 
print(test_csv.info()) # 컴파일 시 결측치 있는걸 확인할 수 있음

############### 결측치(non-null) 처리 2. 평균값 넣기 ###############
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info()) 
print(test_csv.shape)   #(715, 9)

#2. 모델구성
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=9))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 15,    
    restore_best_weights = True,
)

hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=1, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

"""
print("=============== history ====================")
print(hist) # 랩핑 데이터 출력
print("=============== hist.history ====================")
print(hist.history) # 딕셔너리 형태로 출력됨
print("=============== loss ====================")
print(hist.history['loss'])
print("=============== val_loss ====================")
print(hist.history['val_loss'])
"""

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') 
plt.title('따릉이 loss')
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.grid()
plt.show()

"""
loss =  0.4268846809864044
"""