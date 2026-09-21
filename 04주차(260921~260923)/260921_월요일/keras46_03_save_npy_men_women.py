# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/code

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1. 데이터

path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_data/man_woman/'

train_datagen = ImageDataGenerator(
    rescale=1./255, 
)

test_datagen = ImageDataGenerator(   
    rescale=1./255,
)

xy = train_datagen.flow_from_directory(
    path,     
    target_size=(150,150),  
    batch_size=30000,      
    class_mode='binary', 
    color_mode='rgb',  
    shuffle=True,
)
x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     
    random_state=1024,
    stratify=y,
)

np_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
np.save(np_path + 'man_woman_x_train.npy', arr=x_train) # xy_train[0][0] 대신 x_train 가능
np.save(np_path + 'man_woman_y_train.npy', arr=y_train)    # y_train
np.save(np_path + 'man_woman_x_test.npy', arr=x_test)     # x_test
np.save(np_path + 'man_woman_y_test.npy', arr=y_test)     # y_test

print(x_train.shape, y_train.shape) # (21733, 150, 150, 3) (21733,)
print(x_test.shape, y_test.shape)   # (5434, 150, 150, 3) (5434,)
