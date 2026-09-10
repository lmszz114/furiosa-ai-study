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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x)
print(np.min(x), np.max(x))
# 0.0 1.0000000000000002
# 왜 0~1 사이라고 했는데 max 값이 1.00...2 가 나왔을까?
# 파이썬 부동소수점 연산의 문제라서 신경안써도 됨 -> 1이라고 생각하면 됨


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=2048,
)

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
def RMSE(y_test, y_predict):    # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt << 루트 씌우기
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
loss =  0.5438182950019836
RMSE =  0.7374404524838729
"""