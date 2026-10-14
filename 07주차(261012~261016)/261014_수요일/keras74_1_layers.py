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