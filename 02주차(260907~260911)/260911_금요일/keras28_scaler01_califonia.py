# keras19_overfit1_califonia 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

"""
MinMaxScaler

원값 - min
ㅡㅡㅡㅡㅡㅡ
Max - Min

이 알고리즘으로 된 기능을 사이킷런에서 제공함
"""
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)
print(np.min(x), np.max(x))
# MinMaxScaler
# 여기까진 문제가 없으나, 아래 train_test_split 에서 문제가 있음
# 문제가 뭔지 들어도 이해 잘 안감
# train과 test가 이미 과적합 안에 들어가있고, 그래서 train을 따로 스케일링 해야한다는건 알겠음
# 그래서 스케일을 train_test_split 밑으로 내려줘야함
"""

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=2048,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) # 아래처럼 줄여서 한줄에 쓸 수도 있음
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
# 0.0 1.0000000000000002
# -0.00030304199060018824 1.0161943319838045



#2. 모델구성
model = Sequential()
model.add(Dense(40,input_dim=8))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, verbose=1, epochs=1000, batch_size=32, validation_split=0.2)

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)


print("=============== loss ====================")
print(hist.history['loss'])
print("=============== val_loss ====================")
print(hist.history['val_loss'])
print("=============== loss, rmse ====================")
print('loss = ', loss)
print("RMSE = ", rmse)

"""
=============== loss, rmse ====================
loss =  0.5416038036346436
RMSE =  0.7359374176308916
"""

"""
StandardScaler 
loss =  0.5370374321937561
RMSE =  0.7328283990471017
비슷함
"""

"""
MaxAbsScaler
loss =  0.5488117337226868
RMSE =  0.7408183177958041
비슷함
"""

"""
RobustScaler
loss =  0.5370853543281555
RMSE =  0.7328611690847804
비슷함
"""