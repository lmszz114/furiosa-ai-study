import pandas as pd
import numpy as np
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

######## 스케일링1 ########
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1), padding='same'))  
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', padding='same')) 
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', padding='same')) 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))  
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
    patience=30,
    restore_best_weights=True
    )

model.fit(x_train, y_train, epochs=40, batch_size=32, 
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
GPU
loss:  0.30086490511894226
acc:  0.9035999774932861
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.9036
훈련 시간:  164.1 초
"""

"""
MaxPool, padding, stride 등 활용하여 점수 갱신
loss:  0.2800102233886719
acc:  0.90420001745224
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.9042
훈련 시간:  179.02 초
"""