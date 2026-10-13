# 71_2 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,2,3,4,5,])
y = np.array([1,2,3,4,5,])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

################ 동결 ################
# model.trainable = False     # 동결 ★★★★★
model.trainable = True
################ 동결 ################
model.summary()

print("================================")
print(model.weights)
print("================================")

exit()

# [실습]
# weights로 손계산해서 1을 찾아보기