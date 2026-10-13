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
print("==================== model.weight ======================")
print(model.weights)
print("==================== model.trainable weights ======================")
print(model.trainable_weights)
print("=============================")
print(len(model.weights))
print(len(model.trainable_weights))
