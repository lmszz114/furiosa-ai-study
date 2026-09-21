"""
개, 고양이 가중치를 가져와서 모델 완성 
데이터는 개, 고양이 npy 데이터로 사용
내 사진도 npy 불러와서 predict만 하면 됨
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + "catdog_x_train.npy")
y_train = np.load(np_path + "catdog_y_train.npy")
x_test = np.load(np_path + "catdog_x_test.npy")
y_test = np.load(np_path + "catdog_y_test.npy")

#2. 모델구성
load_path = 'C:/study/_save/temp/'
model = load_model(load_path + 'catdog.keras')

#3. 컴파일, 훈련

#4. 평가, 예측
pred_me = np.load(np_path + "keras48_me.npy")
y_predict = model.predict(pred_me)
print(y_predict)    # [[0.67407054]]