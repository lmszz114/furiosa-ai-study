# 2026.09.18 복습 — 함수형 변환 · ImageDataGenerator(폴더 이미지·증강) · npy 저장/불러오기

> 오늘 흐름: CNN 모델을 함수형으로 변환 → 폴더에 있는 이미지 파일을 직접 불러와 수치화(ImageDataGenerator·flow_from_directory)하고 데이터를 증강 → 매번 수치화하는 게 느려서 넘파이(.npy)로 저장해두고 다음부터 불러와 재사용.

---

## 1. keras43_hamsu01~04 — CNN 모델을 함수형으로 변환

mnist·fashion·cifar10·cifar100의 CNN 모델을 Sequential에서 함수형(Functional)으로 바꾸는 실습이다.
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
input1 = Input(shape=(28,28,1))
c1 = Conv2D(64, (3,3))(input1)     # 층을 함수처럼 호출해 연결
...
model = Model(inputs=input1, outputs=output1)
```
- 같은 구조라면 Sequential과 함수형의 파라미터·성능은 동일하다(표현 방식만 다름).

---

## 2. keras44_ImageDataGenerator1 — 폴더에서 이미지 불러오기

지금까지는 mnist처럼 내장된 이미지를 썼다. 이번에는 **내 폴더에 있는 이미지 파일들**을 직접 읽어 수치화한다.

### ImageDataGenerator — 이미지 전처리·증강 도구
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,           # 스케일링 (0~255 → 0~1)
    horizontal_flip=True,     # 좌우 뒤집기
    vertical_flip=True,       # 상하 뒤집기
    width_shift_range=0.1,    # 좌우 이동
    height_shift_range=0.1,   # 상하 이동
    rotation_range=5,         # 회전(각도)
    zoom_range=1.2,           # 확대
    shear_range=0.7,          # 기울이기(전단): 한 좌표 고정하고 다른 좌표를 밀기
    fill_mode='nearest',      # 이동·회전으로 생긴 빈 공간 채우기
)

test_datagen = ImageDataGenerator(rescale=1./255)   # test는 스케일링만
```
- **데이터 증강(augmentation)**: 원본 이미지를 뒤집고·돌리고·이동·확대해 **변형된 이미지를 만들어 데이터 양을 늘린다.** 적은 데이터로도 다양한 형태를 학습해 과적합을 줄이고 일반화를 높인다.
- **test 데이터에는 증강을 하지 않는다.** 평가는 원본 그대로 해야 하므로 `rescale`(스케일링)만 적용한다.

### flow_from_directory — 폴더에서 읽으며 자동 라벨링
```python
path_train = './_data/image/brain/train/'   # train 폴더 (하위: ad, normal)
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),   # 이미지 크기를 통일 (제각각인 이미지를 맞춤)
    batch_size=10,           # 전처리 단계에서 배치 크기 지정
    class_mode='binary',     # 이진분류 (ad / normal)
    color_mode='grayscale',  # 흑백 → 채널 1
    shuffle=True,
)
# Found 160 images belonging to 2 classes.
```
- **경로를 상위 폴더까지만 주면, 하위 폴더별로 자동 라벨링**한다. (`ad` 폴더 → 한 클래스, `normal` 폴더 → 다른 클래스.)
- **`target_size`**: 크기가 다른 이미지들을 지정 크기로 통일한다.
- **`color_mode='grayscale'`**: 흑백으로 읽어 채널이 1이 된다.

### DirectoryIterator 구조 — 배치를 인덱스로 꺼내기
`flow_from_directory`의 결과는 **DirectoryIterator(반복자)** 다. 배치 단위로 x, y가 묶여 있다.
```python
xy_train[0]        # 첫 번째 배치 (x와 y가 tuple로 묶임)
xy_train[0][0]     # 첫 배치의 x (이미지). shape (10, 100, 100, 1)
xy_train[0][1]     # 첫 배치의 y (라벨). shape (10,)
```
- `batch_size=10`이고 이미지가 160장이면 **배치가 16개(0~15)** 생긴다. `xy_train[16]`부터는 없어서 에러가 난다.
- `type` 확인: DirectoryIterator → 안의 배치는 tuple → 그 안의 x·y는 numpy 배열.

---

## 3. keras44_ImageDataGenerator2 / 3 — brain·CatDog 실습

- **ImageDataGenerator2 (brain)**: `batch_size`를 전체 개수로 잡아 한 배치에 다 담고, `xy_train[0][0]`(x), `xy_train[0][1]`(y)를 꺼내 학습한다. 목표 acc 1.0.
- **ImageDataGenerator3 (CatDog)**: 고양이·개 이미지 이진분류. 데이터가 크다.

### 이 실습의 핵심 — 이진분류 세트 확인
brain·CatDog는 이진분류(2클래스)이므로 다음 세트를 지킨다.

| 자리 | 값 |
|---|---|
| 출력층 activation | `sigmoid` |
| compile loss | `binary_crossentropy` |
| 예측 후처리 | `np.round` |

- 이때 y는 `class_mode='binary'`라 원-핫이 아닌 1차원(`[0,1,0,…]`)이므로, 예측은 argmax가 아니라 **round**로 0/1 판정한다.

### 발견된 문제 — 매번 수치화가 느리다
- 이미지가 크거나 많으면(CatDog 등) 실행할 때마다 ImageDataGenerator로 **이미지를 수치화하는 데 시간이 오래 걸린다.**
- 큰 이미지를 큰 batch로 GPU에 한꺼번에 올리면 **메모리 부족(OOM)** 이 날 수 있으므로, 이미지 실습에서는 batch_size를 작게(예: 16) 쓰거나 이미지 크기를 줄인다.
- → 매번 수치화하는 낭비를 없애기 위해 **수치화한 결과를 파일로 저장**한다(아래 4).

---

## 4. keras45 — 수치화한 데이터를 npy로 저장하고 불러오기

이미지를 수치화한 numpy 배열을 **`.npy` 파일로 한 번 저장**해두면, 다음부터는 수치화 과정 없이 **불러오기만** 하면 되어 훨씬 빠르다.

### 저장 (keras45_01_brain_save_npy)
```python
np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr=xy_train[0][0])   # x_train
np.save(np_path + 'keras45_01_y_train.npy', arr=xy_train[0][1])   # y_train
np.save(np_path + 'keras45_01_x_test.npy',  arr=xy_test[0][0])
np.save(np_path + 'keras45_01_y_test.npy',  arr=xy_test[0][1])
exit()   # 저장만 하고 종료 (모델 훈련은 안 함)
```
- **`np.save('경로.npy', arr=배열)`**: 넘파이 배열을 `.npy` 파일로 저장한다.
- 이 스크립트는 이미지 수치화 → 저장까지만 하고 `exit()`으로 끝낸다.

### 불러오기 (keras45_02_brain_load_npy)
```python
np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy')
y_train = np.load(np_path + 'keras45_01_y_train.npy')
x_test  = np.load(np_path + 'keras45_01_x_test.npy')
y_test  = np.load(np_path + 'keras45_01_y_test.npy')
print(x_train.shape, y_train.shape)   # (160, 150, 150, 1) (160,)
# 이후 바로 모델 학습 (ImageDataGenerator 수치화 과정이 없음)
```
- **`np.load('경로.npy')`**: 저장해둔 배열을 불러온다. ImageDataGenerator로 다시 수치화할 필요가 없어 **빠르다.**
- **keras45_03/04 (CatDog)**: 같은 방식으로, 메모리 한계 직전까지 크게 수치화한 데이터를 저장해두고 불러와 학습한다.

---

## 5. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras43_hamsu01~04 | 함수형 변환 | CNN을 Input·Model 함수형으로 |
| keras44_ImageDataGenerator1 | 폴더 이미지 | ImageDataGenerator·flow_from_directory, 증강, DirectoryIterator |
| keras44_ImageDataGenerator2/3 | brain·CatDog | 이진분류 세트, OOM 주의, 매번 수치화 느림 |
| keras45_01~04 | npy 저장/불러오기 | np.save로 저장·np.load로 재사용(빠름) |

---

## 6. 오늘 한 줄 요약

> 폴더에 있는 이미지는 **ImageDataGenerator + flow_from_directory**로 불러오며(하위 폴더로 자동 라벨링, target_size로 크기 통일), 증강 옵션으로 이미지를 변형해 데이터를 늘린다(test는 증강 안 함). 결과는 배치를 인덱스로 꺼내는 DirectoryIterator다.
> 이미지 수치화는 매번 오래 걸리므로, 수치화한 배열을 **`np.save`로 `.npy`에 저장**해두고 다음부터 **`np.load`로 불러와** 재사용하면 훨씬 빠르다.
