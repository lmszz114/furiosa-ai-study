#74_1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.applications import VGG16

vgg16 = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

print(len(model.weights))   # 30 
print(len(model.trainable_weights)) # 4

import pandas as pd
pd.set_option('max_colwidth', None)     # 문자열을 제한없이 보여줌
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]
results = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)
print(results)
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True
