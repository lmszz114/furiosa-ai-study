# 45-1 카피

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

"""
#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    # width_shift_range=0.1,  # 평형 이동
    # height_shift_range=0.1, # 
    # rotation_range=5,       # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,         # 확대
    # shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    # fill_mode='nearest',    # 채우기 (이동하고 비어있는 부분 채우기)
)

test_datagen = ImageDataGenerator(   
    rescale=1./255,
)

path_train = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/brain/train/' 
path_test = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,     
    target_size=(150,150),  
    batch_size=160,      
    class_mode='binary', 
    color_mode='grayscale',  
    shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),
    batch_size=120,    
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=False,
)
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)
"""

np_path = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/npy_data/'
# np.save(np_path + 'keras45_01_x_train.npy', arr=xy_train[0][0]) # xy_train[0][0] 대신 x_train 가능
# np.save(np_path + 'keras45_01_y_train.npy', arr=xy_train[0][1])    # y_train
# np.save(np_path + 'keras45_01_x_test.npy', arr=xy_test[0][0])     # x_test
# np.save(np_path + 'keras45_01_y_test.npy', arr=xy_test[0][1])     # y_test

x_train = np.load(np_path + "keras45_01_x_train.npy")
y_train = np.load(np_path + "keras45_01_y_train.npy")
x_test = np.load(np_path + "keras45_01_x_test.npy")
y_test = np.load(np_path + "keras45_01_y_test.npy")
print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)


exit()


#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150, 150, 1)))  
model.add(Conv2D(16, (3,3), activation='relu')) 
model.add(Conv2D(16, (3,3), activation='relu')) 
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

"""
path = "C:/study/_save/temp/"
model = load_model(path + "keras44_ImageDataGenerator1.keras")
model.summary()
"""


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              metrics=['acc'],
              )


es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=9999,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=4,
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
y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score: ", acc_score)
# print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""

"""