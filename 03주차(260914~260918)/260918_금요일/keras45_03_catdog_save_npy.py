import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


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

path_train = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/cat_dog/train/' 
path_test = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/cat_dog/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,     
    target_size=(155,155),  
    batch_size=10000,      
    class_mode='binary', 
    color_mode='rgb',  
    shuffle=True,
)
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(155,155),
    batch_size=10000,    
    class_mode='binary',
    color_mode='rgb', 
    shuffle=False,
)
# Found 2023 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

np_path = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/npy_data/'
np.save(np_path + 'catdog_x_train.npy', arr=xy_train[0][0]) # xy_train[0][0] 대신 x_train 가능
np.save(np_path + 'catdog_y_train.npy', arr=xy_train[0][1])    # y_train
np.save(np_path + 'catdog_x_test.npy', arr=xy_test[0][0])     # x_test
np.save(np_path + 'catdog_y_test.npy', arr=xy_test[0][1])     # y_test

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)   


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(155, 155, 3), padding='same'))  
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu', padding='same')) 
model.add(Conv2D(16, (3,3), activation='relu', padding='same')) 
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", 
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
model.fit(x_train, y_train, epochs=1000, batch_size=128,
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
