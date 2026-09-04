import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15,16, 9, 6,17,23,20])

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(140))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

print("=============================")


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss = ', loss) # 이것저것 바꿔서 아무리 해봐도 11.3... 미만으로 안떨어짐
result = model.predict(x)
print(result)


# 그래프 그리기
plt.scatter(x, y)
plt.plot(x, result, color='red')
plt.show()