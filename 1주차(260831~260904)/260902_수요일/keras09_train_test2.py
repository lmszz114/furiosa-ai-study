import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#아래는 직접 나누는 방법
# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10]) 
# y_test = np.array([8,9,10])

#[찾아보기] 넘파이 리스트의 슬라이싱 -> 7:3으로 나누자
# x_train = x[0:7] # [1 2 3 4 5 6 7] / 0:7 에서 앞에 0은 생략해도 됨 / 0 생략 시 자동으로 첫 인덱스부터 시작 
# y_train = y[7:] # [ 8  9 10] / 0 생략해서 끝까지 슬라이싱
# print(x_train, y_test)

x_train = x[:7]
y_train = y[:7]
# x_test = x[7:10]
x_test = x[7:]
y_test = y[7:]
print(x_train, y_train, x_test, y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
results = model.predict(np.array([11]))
print("(11)의 예측값: ", results)