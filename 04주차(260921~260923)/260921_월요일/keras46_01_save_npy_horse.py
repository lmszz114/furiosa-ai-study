import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



#1. 데이터

path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_data/horse_human/'

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
print(x_train.shape, x_test.shape)
# (821, 200, 200, 3) (206, 200, 200, 3)

np_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
np.save(np_path + 'horse_x_train.npy', arr=x_train) # xy_train[0][0] 대신 x_train 가능
np.save(np_path + 'horse_y_train.npy', arr=y_train)    # y_train
np.save(np_path + 'horse_x_test.npy', arr=x_test)     # x_test
np.save(np_path + 'horse_y_test.npy', arr=y_test)     # y_test

print(x_train.shape, y_train.shape) #(821, 200, 200, 3) (821,)
print(x_test.shape, y_test.shape)   #(206, 200, 200, 3) (206,)