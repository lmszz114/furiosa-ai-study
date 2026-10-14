# 74_3 카피
# vgg16 부분 동결

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.applications import VGG16

vgg16 = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
)
# vgg16.trainable = False
vgg16.layers[11].trainable = False

# model.layers[2].trainable = False
vgg16.summary()

import pandas as pd
pd.set_option('max_colwidth', None)     # 문자열을 제한없이 보여줌
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in vgg16.layers
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


