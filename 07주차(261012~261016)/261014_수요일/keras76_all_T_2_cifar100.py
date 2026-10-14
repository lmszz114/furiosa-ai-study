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

model_list = [VGG19(include_top=False, input_shape=(32, 32, 3)), 
              ResNet50(include_top=False, input_shape=(32, 32, 3)), 
              ResNet101(include_top=False, input_shape=(32, 32, 3)), 
              DenseNet201(include_top=False, input_shape=(32, 32, 3)), 
            #   InceptionV3(include_top=False, input_shape=(32, 32, 3)), 
            #   InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
              MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
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

from tensorflow.keras.datasets import cifar10, cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer="adam", metrics=["acc"])

    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=======================================")
    print("모델명: ", model_i.name, "loss: ", loss[0], "acc", loss[1])