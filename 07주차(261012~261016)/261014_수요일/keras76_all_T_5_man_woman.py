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



model_list = [VGG19(include_top=False, input_shape=(100, 100, 3)), 
              ResNet50(include_top=False, input_shape=(100, 100, 3)), 
              ResNet101(include_top=False, input_shape=(100, 100, 3)), 
              DenseNet201(include_top=False, input_shape=(100, 100, 3)), 
            #   InceptionV3(include_top=False, input_shape=(32, 32, 3)), 
            #   InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
              MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
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