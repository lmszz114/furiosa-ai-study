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



model_list = [VGG19(include_top=False, input_shape=(200, 200, 3)), 
              ResNet50(include_top=False, input_shape=(200, 200, 3)), 
              ResNet101(include_top=False, input_shape=(200, 200, 3)), 
              DenseNet201(include_top=False, input_shape=(200, 200, 3)), 
            #   InceptionV3(include_top=False, input_shape=(32, 32, 3)), 
            #   InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
              MobileNetV2(include_top=False, input_shape=(200, 200, 3)),
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

path = './_data/image/horse_human/'
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
    target_size=(200,200),  
    batch_size=10000,      
    class_mode='binary', 
    color_mode='rgb',  
    shuffle=True,
)
x = xy[0][0]
y = xy[0][1]
print(x.shape, y.shape)
# (1027, 200, 200, 3) (1027,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     
    random_state=1024,
    stratify=y,
)

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