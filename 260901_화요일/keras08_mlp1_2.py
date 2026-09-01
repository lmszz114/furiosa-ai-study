import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]]) # 잘못된 쉐이프 (2,5) 나옴 / 아래에서 행렬을 변환해야함
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]]) # 의도대로 맞으려면 이렇게 해야함 (5,2)
x = x.T # x 배열의 행렬을 변환함 (행과 열을 바꿈)
x = x.transpose # 위에꺼랑 둘중에 하나 선택해서 쓰면 됨
y = np.array([1,2,3,4,5]) 

print(x.shape)  # (5,2)
print(y.shape)  # (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))    # dim의 개수는 데이터 컬럼(열) 개수와 같음
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)
results = model.predict(np.array([[6,11]]))
print("(6, 11)의 예측값: ", results)