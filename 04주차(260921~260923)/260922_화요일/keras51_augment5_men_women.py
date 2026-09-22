# 여자 데이터 증폭해서 성능 올리기

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
path = 'C:/study/_save/keras46/'
x_train = np.load(path + "men_women_x_train.npy")
y_train = np.load(path + "men_women_y_train.npy")
x_test = np.load(path + "men_women_x_test.npy")
y_test = np.load(path + "men_women_y_test.npy")

x_train_woman = x_train[np.where(y_train > 0.0)]    # 0.0 보다 큰 y값이 있는 인덱스
y_train_woman = y_train[np.where(y_train > 0.0)]

print(x_train_woman.shape, y_train_woman.shape) # (7591, 100, 100, 3) (7591,)
print(np.unique(y_train_woman, return_counts=True)) # (array([1.], dtype=float32), array([7591], dtype=int64))

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train,
    train_size=0.8,     
    random_state=1024,
)

datagen = ImageDataGenerator(
    # rescale=1./255, 
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    # vertical_flip=True,     # 수직 뒤집기 (상하반전)
    width_shift_range=0.1,  # 평형 이동
    #height_shift_range=0.1, # 
    rotation_range=15,       # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=0.1,         # 확대
    #shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 채우기 (이동하고 비어있는 부분 채우기)
)

augment_size = 8000
randidx = np.random.choice(x_train_woman.shape[0], size=augment_size)
x_augmented = x_train_woman[randidx].copy()
y_augmented = y_train_woman[randidx].copy()

print(x_augmented.shape)    # (8000, 100, 100, 3)
print(y_augmented.shape)    # (8000,)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (25386, 100, 100, 3) (25386,)


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (7,7), input_shape=(100, 100, 3),))  
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(64, (7,7), activation='relu',))
model.add(Dropout(0.2))
model.add(MaxPooling2D()) 
model.add(Conv2D(32, (5,5), activation='relu',)) 
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              metrics=['acc'],
              )

model_path = 'C:/study/_save/keras46/'
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=50,
    restore_best_weights=True
    )
mcp = ModelCheckpoint(monitor='val_acc', mode='auto', save_best_only=True, filepath=model_path + 'men_women.keras', verbose=1,)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es,mcp],
          )
end_time = time.time()

#4. 평가, 예측
print("==================model.evaluate==================")
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=32)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(x_test, batch_size=32)
y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print("acc: ", round(acc_score, 4))
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
이전 성능
loss:  0.2437417358160019
acc:  0.9035701155662537
170/170 [==============================] - 1s 4ms/step
acc:  0.9036
훈련 시간:  305.6 초
"""

"""
여자 데이터만 증폭 시켜서 성능 개선 테스트
loss:  0.4266544580459595
acc:  0.8419599533081055
136/136 [==============================] - 1s 4ms/step
acc:  0.842
훈련 시간:  1127.22 초
"""