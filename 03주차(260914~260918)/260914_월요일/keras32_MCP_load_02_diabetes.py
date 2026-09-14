from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_diabetes() # (442, 10) (442,)
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.5,
    random_state=32,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#2. 모델구성
model = load_model("C:/study/_save/keras31/k31_0914_1414-0062-3578.8127.keras")

#3. 컴파일, 훈련

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)


"""
loss =  2980.1181640625
"""
