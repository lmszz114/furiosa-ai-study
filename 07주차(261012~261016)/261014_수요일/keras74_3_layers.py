# 74_2 카피
# layer 부분 동결

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
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable = False
model.summary()


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
# 2       Dense      dense            False
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True


