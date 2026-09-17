import pandas as pd
import numpy as np
import time
from keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder


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

x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)


ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)


#2. 모델구성
model = Sequential()
model.add(Dense(124, input_shape=(32*32*3,), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(124, activation='relu')) 
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='softmax'))  
model.summary()


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", 
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

model.fit(x_train, y_train, epochs=200, batch_size=256, 
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
y_predict = np.argmax(y_predict, axis=1) 
y_test = np.argmax(y_test, axis=1) 

acc_score = accuracy_score(y_test, y_predict) 
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
GlobalAveragePooling2D 사용해서 성능 향상시키기
loss:  2.939432144165039
acc:  0.2540000081062317
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.254
훈련 시간:  465.78 초
"""

"""
DNN 모델로 재구성해서 테스트
- CPU

- GPU
loss:  3.550935983657837
acc:  0.15139999985694885
313/313 [==============================] - 0s 688us/step
accuracy_score:  0.1514
훈련 시간:  54.84 초
"""