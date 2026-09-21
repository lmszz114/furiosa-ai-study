"""
> 여자, 남자 이미지 데이터 실습
keras46_03_save_npy_men_women.py 
 -> 데이터 넘파이 저장
keras47_03_load_npy_men_women.py 
 -> 넘파이 데이터 불러와서 훈련 후 가중치 저장
keras49_02_내가남자게여자게.py 
 -> 넘파이, 가중치 불러와서 내 사진으로 predict
keras48_img_to_array.py 
 -> 재활용, 내사진
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
np_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
x_train = np.load(np_path + "man_woman_x_train.npy")
y_train = np.load(np_path + "man_woman_y_train.npy")
x_test = np.load(np_path + "man_woman_x_test.npy")
y_test = np.load(np_path + "man_woman_y_test.npy")


#2. 모델구성
model_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
model = load_model(model_path + 'man_woman.keras')

#3. 컴파일, 훈련

#4. 평가, 예측
pred_me = np.load(np_path + "lms.npy")
me_predict = model.predict(pred_me)
print(me_predict)    # [[0.13023703]]    # 0에 가까울수록:남자, 1에 가까울수록:여자