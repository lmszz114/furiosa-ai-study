from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input, LSTM
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True,
    width_shift_range=0.1, rotation_range=15, fill_mode='nearest',
)

augment_size = 40000
randidx = np.random.choice(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()     # (40000, 28, 28)
y_augmented = y_train[randidx].copy()

# ★ 증폭 전 4차원으로 (flow는 4차원 필요)
x_augmented = x_augmented.reshape(-1, 28, 28, 1)
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size, shuffle=False,
).next()[0]                               # (40000, 28, 28, 1), rescale로 이미 /255

# 원본도 4차원 + 스케일
x_train = x_train.reshape(-1, 28, 28, 1) / 255.
x_test  = x_test.reshape(-1, 28, 28, 1) / 255.

# 합치기 (둘 다 4차원이라 OK)
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape)   # (100000, 28, 28, 1)

# ★ 마지막에 LSTM용 (N, 784, 1)로 변환
x_train = x_train.reshape(-1, 28*28, 1)
x_test  = x_test.reshape(-1, 28*28, 1)
print(x_train.shape, x_test.shape)   # (100000, 784, 1) (10000, 784, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, input_shape=(784,1)))
model.add(Conv1D(128, 2))
model.add(GlobalAveragePooling1D())
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))  
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5  # 0.5=반띵
)

start_time = time.time()

es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=30,
    restore_best_weights=True
    )

model.fit(x_train, y_train, epochs=10, batch_size=1000, 
          verbose=1, 
          validation_split=0.2,
          callbacks=[es,rlr], 
          )
end_time = time.time()


#4. 평가, 예측
print("==================model.evaluate==================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
#y_test = np.argmax(y_test, axis=1) 

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
LSTM으로 재구성
loss:  2.302828550338745
acc:  0.10000000149011612
313/313 [==============================] - 4s 13ms/step
accuracy_score:  0.1
훈련 시간:  42.08 초
"""

"""
Conv1D 적용
loss:  1.9633917808532715
acc:  0.26460000872612
313/313 [==============================] - 0s 917us/step
accuracy_score:  0.2646
훈련 시간:  67.0 초
"""