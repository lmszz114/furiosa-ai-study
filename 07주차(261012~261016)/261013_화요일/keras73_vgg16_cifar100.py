import pandas as pd
import numpy as np
import time
from keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.applications import VGG16


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0


######## 스케일링1 ########
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)


#2. 모델구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3),
)

# vgg16.trainable = True # 모든 가중치 학습시킨다는 뜻      # 동결X
vgg16.trainable = False # 특징 추출 가중치는 학습 하지 않음     # 동결O

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))
# 실습 # 
# 비교할거
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하고 훈련시켰을때, trainable=False
# 3. 가중치를 동결하지 않고 훈련시켰을때, trainable=True
# 4. 시간까지 비교


#3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="adam", 
              metrics=['acc'],
              )

start_time = time.time()

es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=999,
    restore_best_weights=True
    )

model.fit(x_train, y_train, epochs=60, batch_size=550, 
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
#y_test = np.argmax(y_test, axis=1) # .reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)   # 원값과 예측값
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
loss:  2.7241950035095215
acc:  0.3463999927043915
313/313 [==============================] - 1s 4ms/step
accuracy_score:  0.3464
훈련 시간:  232.63 초
"""