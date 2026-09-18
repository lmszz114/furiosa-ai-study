import pandas as pd
import numpy as np
import time
from keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
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

x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)


ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)


#2. 모델구성
# model = Sequential()
# model.add(Conv2D(64, (5,5), input_shape=(32, 32, 3), padding='same'))  
# model.add(Dropout(0.2))
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3), activation='relu', padding='same')) 
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), activation='relu', padding='same')) 
# model.add(Dropout(0.2))
# # model.add(Flatten())
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=32, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100, activation='softmax'))  
# model.summary()


###########################################################

#2-2. 함수형 모델
input1 = Input(shape=(32,32,3))
conv1 = Conv2D(64, (3,3), padding='same')(input1)
drop1 = Dropout(0.2)(conv1)
conv2 = Conv2D(32, (3,3), activation='relu')(drop1)
drop2 = Dropout(0.2)(conv2)
maxpool1 = MaxPooling2D()(drop2)
conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(maxpool1)
drop3 = Dropout(0.2)(conv3)
maxpool2 = MaxPooling2D()(drop3)
conv4 = Conv2D(32, (3,3), activation='relu', padding='same')(maxpool2)
drop4 = Dropout(0.2)(conv4)
maxpool3 = MaxPooling2D()(drop4)
conv5 = Conv2D(32, (3,3), activation='relu', padding='same')(maxpool3)
drop5 = Dropout(0.2)(conv5)
gap1 = GlobalAveragePooling2D()(drop5)
dense1 = Dense(32, activation='relu')(gap1)
drop6 = Dropout(0.2)(dense1)
dense2 = Dense(32, activation='relu')(drop6)
drop7 = Dropout(0.2)(dense2)
dense3 = Dense(32, activation='relu')(drop7)
drop8 = Dropout(0.2)(dense3)
output1 = Dense(100, activation='softmax')(drop8)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

#3. 컴파일, 훈련
model2.compile(loss="categorical_crossentropy", 
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

model2.fit(x_train, y_train, epochs=200, batch_size=512, 
          verbose=1, 
          validation_split=0.2,
          callbacks=[es], 
          )
end_time = time.time()


#4. 평가, 예측
print("==================model.evaluate==================")
loss = model2.evaluate(x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model2.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) 
y_test = np.argmax(y_test, axis=1) 

acc_score = accuracy_score(y_test, y_predict) 
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")


"""
모델을 함수형으로 바꾸고 돌려보기
loss:  2.7785511016845703
acc:  0.2831000089645386
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.2831
훈련 시간:  581.99 초
"""