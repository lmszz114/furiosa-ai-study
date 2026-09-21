from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_data/'

img = load_img(path + "lms.jpg", target_size=(150,150))  #load_img 는 한장짜리 이미지 불러올 때 편함

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
arr = arr/255
print(arr)
print(arr.shape)    # (1, 100, 100, 3)

np_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
np.save(np_path + 'lms.npy', arr=arr) # 내 사진을 넘파이로 변환해서 지정한 경로에 저장