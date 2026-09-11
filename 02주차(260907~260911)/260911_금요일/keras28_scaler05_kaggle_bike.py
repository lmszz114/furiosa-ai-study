import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [10886 rows x 11 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) # [6493 rows x 8 columns]
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission)   # [6493 rows x 1 columns]

print(train_csv.shape, test_csv.shape, submission.shape) # train (10886, 11) / test (6493, 8) / submission (6493, 1)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) # 묘사

######################### 결측치 확인 #########################
print(train_csv.isna().sum()) # 결측치 수치(isna)를 더하기(sum)
print(train_csv.isnull().sum()) # 결측치 위치 찾아서(isnull)를 더하기(sum)


######################### x, y 분리 #########################
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']
print(y, y.shape)    # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=42,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=8))
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
                 epochs=100, 
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
plt.title('캐글 loss')
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.grid()
plt.show()

"""
loss =  22632.509765625

loss =  22260.138671875
별차이 없음
"""

"""
StandardScaler 
loss =  21770.662109375
비슷함
"""

"""
MaxAbsScaler
loss =  22390.74609375
"""

"""
RobustScaler

"""