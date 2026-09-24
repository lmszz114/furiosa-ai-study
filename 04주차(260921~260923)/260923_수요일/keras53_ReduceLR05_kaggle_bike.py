import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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

#2. 순차적 모델
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=8))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
model.summary()
###########################################################

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001     # 0.001 - 디폴트
#learning_rate = 0.0001
#learning_rate = 0.005
#learning_rate = 0.05

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

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

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=100, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es,rlr],
                 )
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
러닝레이트 적용 (미갱신)
learning_rate = 0.01
loss =  26742.40625
걸린시간 =  39.9 초

learning_rate = 0.0001
loss =  25836.046875
걸린시간 =  40.19 초
"""

"""
ReduceLROnPlateau 적용
learning_rate = 0.01
loss =  33292.75
걸린시간 =  40.21 초
"""