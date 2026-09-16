# 36_2 카피

import pandas as pd
import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

######## 스케일링1 ########
x_train = x_train/255.  # 255. float
x_test = x_test/255.
# 이미지 데이터기 때문에 최대 범위가 255로 이미 고정되어있기 때문에 MinMax 안쓰고 255로 나눔
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0
# 범위: 0 ~ 1 사이가 됨 / MinMax 기능과 같음


######## 스케일링2 ########
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

# 범위: -1 ~ 1 사이가 됨 / MaxAbs 기능과 같음
# 통상적으로 이 두가지를 많이 씀

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1) / 4차원 데이터로 리쉐이프

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(60000,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
# y_train, y_test 리쉐이프, 트랜스폼

print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))   # input_shape는 x의 행데이터를 제외한 컬럼값만 넣어줌 # (26,26,64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # 필터와 커널사이즈를 정확히 명시하여 구분 # (24,24,32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu')) # (23, 23, 32)
model.add(Conv2D(16, (2,2), activation='relu')) # (22, 22, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu')) # (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu')) # (20, 20, 16)
# (N, 20, 20, 16) / 필요한 데이터 형식(N, 10)
# (N, 20, 20, 16) 를 (N, 20x20x16) 로 표현 가능 -> (None, 6400) / reshape 가능 (Flatten 사용)

model.add(Flatten())    # 위에서 나온 Output Shape 를 펼쳐서 연결 / (None, 6400)

model.add(Dense(units=32, activation='relu')) # Dense 레이어를 여러개 두어 순차적으로 줄어들며 계산되도록 구성
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))  
model.summary()
# Output Node 의 개수 / 4차원: filters, 2차원: units

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=['acc'],
              )

start_time = time.time()

es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=30,
    restore_best_weights=True
    )

model.fit(x_train, y_train, epochs=200, batch_size=128, 
          verbose=1, 
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()


#4. 평가, 예측
print("==================model.evaluate==================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) # .reshape(-1, 1)
y_test = np.argmax(y_test, axis=1) # .reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)   # 원값과 예측값
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
CPU
loss:  0.051453642547130585
acc:  0.9886999726295471
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
accuracy_score:  0.9887
훈련 시간:  536.41 초

GPU
loss:  0.05006885901093483
acc:  0.9891999959945679
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.9892
훈련 시간:  138.7 초
"""

"""
파라미터 튜닝해서 GPU 성능 올리기 실습: acc 0.995 목표


"""