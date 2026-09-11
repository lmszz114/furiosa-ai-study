# keras29_3 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

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
# model = Sequential()
# model.add(Dense(40, activation='relu', input_dim=8))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1))

# model.summary()

path = 'H:/furiosa-ai-study/02주차(260907~260911)/260911_금요일/_save/'
model = load_model(path + 'keras29_3_save_model.keras')
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, verbose=1, epochs=10, batch_size=32, validation_split=0.2)

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

# model.save(path + 'keras29_3_save_model.keras')


print("=============== loss ====================")
print(hist.history['loss'])
print("=============== val_loss ====================")
print(hist.history['val_loss'])
print("=============== loss, rmse ====================")
print('loss = ', loss)
print("RMSE = ", rmse)

"""
RobustScaler
loss =  0.5370853543281555
RMSE =  0.7328611690847804
비슷함
"""