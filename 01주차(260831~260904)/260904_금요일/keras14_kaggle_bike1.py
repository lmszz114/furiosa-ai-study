# https://www.kaggle.com/competitions/bike-sharing-demand/data

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
    random_state=7777,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100,activation='relu', input_dim=8)) # relu 액티베이션 적용 -> 음수를 0으로 변환
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='relu'))   # activation 넣어도 상관없긴한데, 통상적으로 아웃풋에는 안씀, 다만 여기선 안넣으니까 음수 나와서 적용해서 해봄


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4)

print("===================================")


#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

def RMSE(y_test, y_predict):    # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt << 루트 씌우기
rmse = RMSE(y_test, y_predict)

print("RMSE = ", rmse)

############################ submission.csv 만들기 // count 컬럼에 값 넣어준다. ############################
# print(submission)
y_submit = model.predict(test_csv)
submission['count'] = y_submit
# print(submission)
# print(submission.shape)

submission.to_csv(path + "submit/" + "submit_bike_01.csv")

"""
loss =  22599.005859375
RMSE =  150.3296637809717
"""