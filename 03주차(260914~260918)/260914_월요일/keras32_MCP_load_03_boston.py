from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = load_model("C:/study/_save/keras31/k33_0914_1416-0067-10.5349.keras")

#3. 컴파일, 훈련

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

"""
loss =  23.633453369140625
"""
