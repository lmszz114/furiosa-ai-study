# 36_3 카피

import pandas as pd
import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, Activation
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
model.add(Conv2D(filters=32, kernel_size=(3,3), activation=None)) # 필터와 커널사이즈를 정확히 명시하여 구분 # (24,24,32)
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (2,2), activation=None)) # (23, 23, 32)
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(16, (2,2), activation='relu')) # (20, 20, 16)
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

model.fit(x_train, y_train, epochs=50, batch_size=128, 
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
loss:  0.09734233468770981
acc:  0.9879999756813049
313/313 [==============================] - 1s 1ms/step
accuracy_score:  0.988
훈련 시간:  138.9 초
"""