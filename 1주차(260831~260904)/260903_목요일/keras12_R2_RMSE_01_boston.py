from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
# print(y_train.shape, y_test.shape) # (404,) (102,)


#2. 모델구성
model = Sequential()
model.add(Dense(50,input_dim=13))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=5)

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss(mse) = ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 = ", r2) # r2 =  0.7057315795488385
# r2 0.75 넘겨보기

mse = mean_squared_error(y_test, y_predict)
print ("mse = ", mse)

def RMSE(y_test, y_predict):    # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt << 루트 씌우기

rmse = RMSE(y_test, y_predict)
print("RMSE = ", rmse)

# mse =  56.843943155730685
# RMSE =  7.539492234609084