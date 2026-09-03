# [실습]
# R2 기준 0.55 이상 만들어보기

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

# 훈련용, 결과용 나누고 실습

x_train, x_test, y_train, y_test = train_test_split( #x트레인,x테스트->y트레인,y테스트 순서 지켜야함, 함수 안에 x,y 순서대로 매칭)
    x, y,
    train_size=0.8,
    random_state=999,
)

#2. 모델구성
model = Sequential()
model.add(Dense(50,input_dim=8))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1500, batch_size=47)

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print ("r2 = ", r2)

mse = mean_squared_error(y_test, y_predict)
print ("mse = ", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE =", rmse)

# r2 =  0.5823932275407497