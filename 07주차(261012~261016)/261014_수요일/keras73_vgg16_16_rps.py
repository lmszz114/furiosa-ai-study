import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



#1. 데이터

path = './_data/image/rps/'

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

xy = train_datagen.flow_from_directory(
    path,     
    target_size=(150,150),  
    batch_size=10000,      
    class_mode='categorical', 
    color_mode='rgb',  
    shuffle=True,
)
x = xy[0][0]
y = xy[0][1]
print(x.shape, y.shape)
#(2048, 200, 200, 3) (2048, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     
    random_state=1024,
    stratify=y,
)
print(x_train.shape, x_test.shape)
# (1638, 200, 200, 3) (410, 200, 200, 3)


np_path = 'C:/study/_save/keras46/'
np.save(np_path + 'rps_x_train.npy', arr=x_train) # xy_train[0][0] 대신 x_train 가능
np.save(np_path + 'rps_y_train.npy', arr=y_train)    # y_train
np.save(np_path + 'rps_x_test.npy', arr=x_test)     # x_test
np.save(np_path + 'rps_y_test.npy', arr=y_test)     # y_test

print(x_train.shape, y_train.shape) #(1638, 200, 200, 3) (1638, 3)
print(x_test.shape, y_test.shape)   #(410, 200, 200, 3) (410, 3)


#2. 모델구성
from keras.applications import VGG16
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3),
)
# vgg16.trainable = True # 모든 가중치 학습시킨다는 뜻      # 동결X
vgg16.trainable = False # 특징 추출 가중치는 학습 하지 않음     # 동결O
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=['acc'],
              )


es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=50,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=128,
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
print("acc: ", round(acc_score, 4))
print("훈련 시간: ", round(end_time - start_time, 2), "초")


"""
동결O
loss:  0.1658025085926056
acc:  0.9439024329185486
13/13 [==============================] - 1s 50ms/step
acc:  0.9415
훈련 시간:  11.03 초

"""