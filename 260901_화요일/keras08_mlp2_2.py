import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # 0부터 10-1 까지의 정수 -> 0~9 (데이터 10개)
print(x)    # [0 1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 10)) # 1부터 10-1 까지의 정수 -> 1~9 (데이터 9개)
print(x)    # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11)) # 1부터 11-1 까지의 정수 -> 1~10 (데이터 10개)
print(x)    # [ 1  2  3  4  5  6  7  8  9 10]

# x = np.array(range(10), range(21, 31), range(201, 210)) #에러, 데이터 개수 맞지 않음 (9개, 9개, 8개)
x = np.array([range(10), range(21, 31), range(201, 211)]).T #마지막에 .T로 행렬 변환함
print(x.shape)  # (3, 10) -> (10, 3)

y = np.array(range(1, 11))
print(y.shape)  # (10,)

# [실습]
# [10, 31, 211] 찾기
# 11.00.. 나오면 통과

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))    # dim의 개수는 데이터 컬럼(열) 개수와 같음
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=3)


#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss) 
results = model.predict(np.array([[10, 31, 211]]))
print("(10, 31, 211)의 예측값: ", results)  # (10, 31, 211)의 예측값:  [[10.985952]]