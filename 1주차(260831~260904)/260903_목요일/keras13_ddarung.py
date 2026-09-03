# https://dacon.io/competitions/open/235576/data
# 평가방식은 RMSE 수치

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
# path = "./_data/ddarung/"
path = "H:/furiosa-ai-study/1주차(260831~260904)/260903_목요일/" # 집이라서 경로 바꿨음
# 땡겨온 데이터를 수치화 해야함 -> 판다스(pandas) 사용 / 판다스 설치 필요

train_csv = pd.read_csv(path + "train.csv", index_col=0) # index_col=0 첫번째 컬럼을 인덱스로 씀
print(train_csv) 
# 열 포함 [1459 rows x 11 columns]
# 열 미포함 [1459 rows x 10 columns]
# count 를 찾는게 목표인데 id 컬럼이 필요한가?
# id 컬럼은 index 역할만 하고 있을뿐이라 유효한 데이터가 아님
# 판다스 함수 내에서 index_col 파라미터를 사용해서 처리
# 원 데이터 11컬럼에서 index_col 사용 후 10컬럼으로 변경됨

test_csv = pd.read_csv(path + "test.csv", index_col=0) #test.csv 에는 count 컬럼이 없음 -> y값이 없다는 말임
print(test_csv) #test.csv 는 제출용임
# [715 rows x 9 columns]
# 이 테스트 데이터로 predict 시킴

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)
# [715 rows x 1 columns]

# 3개의 파일을 모두 준비했음
# train_csv 를 x,y 로 분리해야함

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)
# 데이터 안보고 쉐이프만 딱딱딱 찍히게 하려면 .shape 사용 / 결과는 위의 각 주석값과 동일하게 나옴

print(train_csv.columns)
# #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')
# 컬럼명 구성을 볼 수 있음

print(train_csv.info())
print(test_csv.info())

# exit()

############### 결측치(non-null) 처리 1. 삭제 ###############
# 임의로 유추하는 것보다 삭제하는 것이 나은 경우에 사용
# 나이가 50대인 10명의 평균 연봉을 계산할 때 9명은 2000~5000만이지만, 한명이 100억인 경우 이상치 발생
# 한명의 이상치 연봉 데이터를 임의로 유추하는 것은 맞지 않는 방법임
# 이런 경우 삭제하는 것이 낫다고 판단
# 반대로 1000개의 데이터 중 999개가 결측치가 있는 경우 삭제하면 1개의 데이터만 남으므로 삭제는 적합하지 않음 
 
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
    # train_size=0.7,
    random_state=888,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=9))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=5)   # 전체(x,y)가 아니라 훈련 데이터로 학습

print("===================================")


#4. 평가, 예측
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):    # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt << 루트 씌우기

rmse = RMSE(y_test, y_predict)
print("RMSE = ", rmse) # 50.

# 내일 이어서 마무리 할 예정