# keras17_val1_califonia 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=6000,
)

#2. 모델구성
model = Sequential()
model.add(Dense(30,input_dim=8))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, verbose=1, epochs=50, batch_size=32, validation_split=0.2)

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

print("=============== history ====================")
print(hist) # 랩핑 데이터 출력
print("=============== hist.history ====================")
print(hist.history) # 딕셔너리 형태로 출력됨
print("=============== loss ====================")
print(hist.history['loss'])
print("=============== val_loss ====================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트 안넣으면 그래프에서 한글이 깨져보임

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:], c='red', label='loss') # loss 값을 두번째부터 보여줌
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss') # loss 값을 두번째부터 보여줌
plt.legend(loc='upper right')   # 우측 상단에 라벨 표시
plt.title('캘리포니아 loss') # 제목
plt.xlabel('epoch') # x축 라벨 이름
plt.ylabel('loss')  # y축 라벨 이름
plt.grid()  # 격자 표시 추가
plt.show()