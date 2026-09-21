# 48 카피

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/study/_data/image/'

img = load_img(path + "lhy.jpg", target_size=(100,100))  #load_img 는 한장짜리 이미지 불러올 때 편함

print(img)
# <PIL.Image.Image image mode=RGB size=100x100 at 0x163E76F9660>
print(type(img)) # <class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (100, 100, 3)
print(type(arr))    # <class 'numpy.ndarray'>
arr = np.expand_dims(arr, axis=0) # 차원 확장 / catdog 에 넣을거라서 차원 맞춰줌
# arr = arr/255
print(arr)
print(arr.shape)    # (1, 100, 100, 3)

# np_path = 'C:/study/_save/keras46/'
# np.save(np_path + 'keras49_me.npy', arr=arr) # 내 사진을 넘파이로 변환해서 지정한 경로에 저장

############## 여기부터 증폭 ##############
datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    vertical_flip=True,     # 수직 뒤집기 (상하반전)
    #width_shift_range=0.1,  # 평형 이동
    #height_shift_range=0.1, # 
    rotation_range=60,       # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.1,         # 확대
    #shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 채우기 (이동하고 비어있는 부분 채우기)
)

it = datagen.flow(arr, 
                  batch_size=1,
)
print(it)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x000002596A677F70>

# print(it.next())    # python 3.10 까지 됨
print(next(it))     # 3.11 부터 이렇게 문법 바뀜
print(next(it).shape)   # (1, 100, 100, 3)


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    # batch = it.next()
    batch = next(it)
    print(batch.shape)
    batch = batch.reshape(100,100,3)
    ax[i].imshow(batch)
    ax[i].axis('off')
plt.show()

