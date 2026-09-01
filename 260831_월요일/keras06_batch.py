from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))
# 데이터를 자를수록 훈련 효율 높음

#.3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print("7의 예측값 : ", result)

# epochs 으로 훈련횟수 지정, 레이어를 분할, 노드를 여러개로 생성, batch_size로 데이터를 분할