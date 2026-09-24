from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

############## 여기부터 증폭 ##############
datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    # vertical_flip=True,     # 수직 뒤집기 (상하반전)
    width_shift_range=0.1,  # 평형 이동
    #height_shift_range=0.1, # 
    rotation_range=15,       # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=0.1,         # 확대
    #shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 채우기 (이동하고 비어있는 부분 채우기)
)

augment_size = 40000

print(x_train.shape[0]) # 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 6만개 중에 4만개 랜덤뽑기
print(randidx)
print(randidx.shape)    # 벡터라서 쉐이프 사용 가능
print(len(randidx)) # 튜플, 리스트는 len 으로 확인해야함 (벡터도 되긴함)

print(np.min(randidx), np.max(randidx)) # 0 49998 # 랜덤

x_augmented = x_train[randidx].copy()    # 랜덤으로 뽑은 4만개를 x_augmented 에 넣는다  # 메모리 별도 공간에 새로 할당
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 32, 32, 3) (40000, 1)  # 이제 쉐이프 변환해서 원데이터에 붙여주면됨

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]     # y는 필요없음
############### 변환 완료 ###############
print(x_augmented.shape) 

print(x_train.shape) #(50000, 32, 32, 3)

x_train = x_train/255.
x_test = x_test/255.

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) 

print(np.unique(y_train, return_counts=True)) 

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32, 32, 3), padding='same'))  
model.add(Dropout(0.2))
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
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu')) 
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
# learning_rate = 0.001     # 0.001 - 디폴트
#learning_rate = 0.0001
#learning_rate = 0.005
#learning_rate = 0.05

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

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
    patience=100,
    restore_best_weights=True
    )

model.fit(x_train, y_train, epochs=1000, batch_size=32, 
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
y_test = np.argmax(y_test, axis=1) 

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
러닝레이트 적용
learning_rate = 0.0001
loss:  0.561287522315979
acc:  0.8179000020027161
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.8179
훈련 시간:  6525.37 초
"""

"""
ReduceLROnPlateau 적용
learning_rate = 0.01
loss:  2.3033339977264404
acc:  0.10000000149011612
313/313 [==============================] - 0s 1ms/step
accuracy_score:  0.1
훈련 시간:  863.6 초
"""