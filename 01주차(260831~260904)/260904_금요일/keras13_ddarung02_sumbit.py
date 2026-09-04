# keras13_ddarung01 내용 카피
# https://dacon.io/competitions/open/235576/data
# 평가방식은 RMSE 수치

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
path = "H:/furiosa-ai-study/01주차(260831~260904)/260904_금요일/따릉이/"
# 땡겨온 데이터를 수치화 해야함 -> 판다스(pandas) 사용 / 판다스 설치 필요
"""
# path = "C:/study/_data/ddarung/" # 절대경로 # 슬래시/역슬래시 상관없으나, 예약어 때문에 슬래시 쓰는게 나아보임
# path = "C://study//_data//ddarung//" # 실행 됨
# path = "C:\\study\\_data\\ddarung\\" # 실행 됨
# path = "C:/study//_data/ddarung//" # 실행 됨
# path = "C:\study/_data\\ddarung//"  # 섞어쓰기도 되지만, 가급적 한가지만 쓸것 (섞어쓰는것 권장하지 않음)
"""

train_csv = pd.read_csv(path + "train.csv", index_col=0) # index_col=0 첫번째 컬럼을 인덱스로 씀
print(train_csv) 
# 열 포함 [1459 rows x 11 columns]
# 열 미포함 [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #test.csv 에는 count 컬럼이 없음 -> y값이 없다는 말임
print(test_csv) #test.csv 는 제출용임
# [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)
# [715 rows x 1 columns]

print(train_csv.shape)
print(test_csv.shape)
print(submission.shape)

print(train_csv.columns)
# #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')
# 컬럼명 구성을 볼 수 있음

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
test_csv = test_csv.fillna(test_csv.mean()) #결측치 채우기
print(test_csv.info()) 
print(test_csv.shape)   #(715, 9)


#2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=5)

print("===================================")


#4. 평가, 예측
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):    # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt << 루트 씌우기

rmse = RMSE(y_test, y_predict)
print("RMSE = ", rmse) # 50.

############################ submission.csv 만들기 // count 컬럼에 값 넣어준다. ############################
print(submission)
y_submit = model.predict(test_csv)
submission['count'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0904.csv")






##### 명세서(예시) #####
'''
- 1차시도
random_state = 337
train_size = 0.75
epoch = 50
batch_size = 1 
- 결과
rmse = 37
r2 = 0.66
'''

'''
- 2차시도
random_state = 4007
train_size = 0.7
epoch = 100
batch_size = 1 
- 결과
rmse = 35
r2 = 0.65
'''
