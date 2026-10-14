import warnings
warnings.filterwarnings("ignore")   # 경고메시지 출력하지 않음, 모든 경고를 무시하진 않음, 대부분은 해결됨

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



model_list = [VGG19(include_top=False, input_shape=(155, 155, 3)), 
              ResNet50(include_top=False, input_shape=(155, 155, 3)), 
              ResNet101(include_top=False, input_shape=(155, 155, 3)), 
              DenseNet201(include_top=False, input_shape=(155, 155, 3)), 
            #   InceptionV3(include_top=False, input_shape=(32, 32, 3)), 
            #   InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
              MobileNetV2(include_top=False, input_shape=(155, 155, 3)),
            #  EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

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

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set/'

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

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'catdog_x_train.npy', arr=xy_train[0][0]) # xy_train[0][0] 대신 x_train 가능
np.save(np_path + 'catdog_y_train.npy', arr=xy_train[0][1])    # y_train
np.save(np_path + 'catdog_x_test.npy', arr=xy_test[0][0])     # x_test
np.save(np_path + 'catdog_y_test.npy', arr=xy_test[0][1])     # y_test

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)   

for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일, 훈련
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer="adam", metrics=["acc"])

    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=======================================")
    print("모델명: ", model_i.name, "loss: ", loss[0], "acc", loss[1])