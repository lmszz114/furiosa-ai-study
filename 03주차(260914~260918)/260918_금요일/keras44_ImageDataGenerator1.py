import numpy as np
from keras.preprocessing.image import ImageDataGenerator
print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평형 이동
    height_shift_range=0.1, # 
    rotation_range=5,       # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range=1.2,         # 확대
    shear_range=0.7,        # 좌표 하나를 고정하고 다른 몇개의 좌표를 이동
    fill_mode='nearest',    # 채우기 (이동하고 비어있는 부분 채우기)
)

test_datagen = ImageDataGenerator(   # 테스트 데이터는 변환하지 않음, 스케일링만 적용
    rescale=1./255,
)

path_train = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/brain/train/'   # 여기까지 잡아주면 하위 폴더인 ad, normal 폴더 내의 데이터를 라벨링함
path_test = 'H:/furiosa-ai-study/03주차(260914~260918)/260918_금요일/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,     # 경로
    target_size=(100,100),  # 이미지의 사이즈가 각각 다른 데이터들이 있을수도 있음. 그렇기 때문에 사이즈를 맞추도록 하기 위해 사용함
    batch_size=10,      # 전처리에서 미리 배치사이즈 조절 가능
    class_mode='binary', # 이진분류 (애드노멀과 노멀을 찾으니까)
    color_mode='grayscale',  # 흑백 (최종으로 나오는 데이터가 1 이니까)
    shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10,    
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=False, # test에서는 필요없음
)
# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000002AB1B157FA0>
# Iterator 가 뭔지 인터넷에서 검색해보기
# print(xy_train.next()) # Iterator의 첫번째 보기 # x,y 가 합쳐져 있는 Iterator 형태로 나오는걸 확인할 수 있음
# print(xy_train.next()) # Iterator의 두번째 보기

# print(xy_train[0])    # 첫번째 배치
# print(xy_train[1])    # 두번째 배치
# print(xy_train[2])    # 세번째 배치 ...

# print(xy_train[0][0])   # 첫번째 배치의 x 데이터
# print(xy_train[0][1])   # 첫번째 배치의 y 데이터

print(xy_train[0][0].shape) # (10, 100, 100, 1)
print(xy_train[0][1].shape) # (10,)
print(xy_train[15][0].shape) # (10, 100, 100, 1)
# print(xy_train[16][0].shape)    # 16부턴 에러남 (ad, normal 합쳐서 160장이라서 15까지만 가능) # ad 배치 10개 x 8 + normal 배치 10개 x 8
# for 문을 사용하면 한번에 전부 볼 수 있음

print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> # 리스트와 비슷하지만 수정 안됨
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> # 안에 넘파이가 들어있다는걸 확인
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> # 얘도 안에 넘파이가 들어있다는걸 확인

