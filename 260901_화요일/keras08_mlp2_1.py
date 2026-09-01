import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# 위 데이터를 토대로 모델구성부터 만들어보기
# loss 0.0015 이하면 합격
# [10, 1.3, 0] 예측해보기 10.00... 이면 합격

# print(x.shape)
# print(y.shape)
x = x.T # x 데이터 변환

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))    # dim의 개수는 데이터 컬럼(열) 개수와 같음
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=3)


#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)  # loss :  3.800916026364121e-09
results = model.predict(np.array([[10, 1.3, 0]]))
print("(10, 1.3, 0)의 예측값: ", results)   #(10, 1.3, 0)의 예측값:  [[10.000061]]