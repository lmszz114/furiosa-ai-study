# 50_1 카피

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

############## 여기부터 증폭 ##############
datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    # vertical_flip=True,     # 수직 뒤집기 (상하반전)
    #width_shift_range=0.1,  # 평형 이동
    #height_shift_range=0.1, # 
    rotation_range=60,       # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.1,         # 확대
    #shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 채우기 (이동하고 비어있는 부분 채우기)
)

augment_size = 100
print(x_train.shape)    # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)

# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape)    # (28, 2800)

aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)    # (100, 28, 28, 1)
# 단순 복붙임. 이대로 데이터 갖다 쓰진 않음
# 단순 복붙은 데이터가 같고 개수만 많아진거라 과적합됨

xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=augment_size,    # 통배치
    shuffle=False,
).next()

print(xy_data)
print(type(xy_data))    # <class 'tuple'>

# print(xy_data.shape)      # AttributeError: 'tuple' object has no attribute 'shape' / 튜플엔 shape 가 없다
print(len(xy_data)) # 2 (x, y가 두개니까)    # 튜플에서 확인하려면 len 사용

print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
plt.show()