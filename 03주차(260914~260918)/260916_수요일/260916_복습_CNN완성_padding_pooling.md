# 2026.09.16 복습 — CNN 완성(Conv2D·Flatten) · 데이터셋 수동 설치 · padding·stride·MaxPooling

> 오늘 흐름: MNIST로 CNN 모델을 끝까지 완성(4차원 입력 → Conv2D → Flatten → Dense) → 다른 이미지 데이터셋 실습 → 다운로드가 막힌 CIFAR를 수동 설치 → 이미지 크기를 조절하는 padding·stride·MaxPooling.

---

## 1. keras36_cnn3_mnist — CNN 모델 완성

### ① 이미지를 4차원으로 reshape
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)          # (60000, 28, 28)

x_train = x_train.reshape(-1, 28, 28, 1)   # (60000, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)
```
- **Conv2D는 4차원 입력 (샘플, 높이, 너비, 채널)** 을 받는다. MNIST는 흑백이라 `(60000, 28, 28)`로 채널이 생략돼 있으므로, `reshape`로 끝에 채널 `1`을 붙여 `(60000, 28, 28, 1)`로 만든다.
- `reshape(-1, 28, 28, 1)`의 `-1`은 "그 자리(샘플 수)는 자동 계산"이다.

### ② 스케일링
```python
x_train = x_train / 255.    # 0~1 (MinMax와 같은 효과)
```
- 이미지 픽셀은 0~255로 범위가 고정이라 스케일러 객체 없이 255로 직접 나눈다.

### ③ y 원-핫 인코딩
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))   # (60000, 10)
y_test  = ohe.fit_transform(y_test.reshape(-1,1))
```
- 10개 클래스(0~9)라 다중분류. y를 원-핫으로 `(60000, 10)`으로 만든다.

### ④ 모델 구성 — Conv2D를 지나면 크기가 줄어든다
```python
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))   # (26, 26, 64)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))  # (24, 24, 32)
model.add(Conv2D(32, (2,2), activation='relu'))          # (23, 23, 32)
...
```
- `input_shape=(28,28,1)`은 행(샘플 수)을 제외한 (높이, 너비, 채널)이다.
- Conv2D를 지날 때마다 커널 크기만큼 이미지가 작아진다: 커널 (3,3)이면 −2, (2,2)이면 −1. (28 → 26 → 24 → 23 → …)
- **filters vs units**: 출력 노드 수를 4차원(Conv2D)에서는 `filters`, 2차원(Dense)에서는 `units`라고 부른다.
- `Conv2D(filters=32, kernel_size=(3,3))`처럼 이름을 명시하면 필터 수와 커널 크기를 명확히 구분할 수 있다.

### ⑤ Flatten — 4차원을 2차원으로 펼치기 (핵심)
Conv2D의 출력은 4차원(예: `(N, 20, 20, 16)`)인데, Dense는 2차원 입력을 받는다. 그래서 그 사이에 **Flatten**으로 펼쳐 연결한다.
```python
model.add(Flatten())                       # (N, 20, 20, 16) → (N, 6400)
model.add(Dense(units=32, activation='relu'))
...
model.add(Dense(10, activation='softmax')) # 출력 10개(클래스 수)
```
- **`Flatten`**: 4차원 출력의 높이·너비·채널을 하나로 이어 붙여 1차원으로 만든다. `20 × 20 × 16 = 6400`이 되어 `(N, 6400)`이 된다.
- 이렇게 펼친 뒤 Dense 층들을 거쳐, 마지막 `Dense(10, softmax)`로 다중분류한다.
- **CNN 구조 요약: Conv2D(특징 추출) → Flatten(펼치기) → Dense(분류) → softmax.**

### ⑥ 훈련·평가와 GPU 효과
- 분류이므로 EarlyStopping의 `monitor='val_acc'`(정확도 감시)도 쓸 수 있다.
- 예측은 다중분류대로 `np.argmax`로 판정한다.
- **CPU 536초 vs GPU 138초** — 이미지 + CNN처럼 계산이 무거운 작업은 GPU가 확실히 빠르다.

---

## 2. keras36_cnn4 / 5 / 6 — 이미지 데이터셋 실습

데이터셋만 제시되고 나머지는 위 CNN 형태로 직접 구현하는 실습이다.

| 파일 | 데이터셋 | 클래스 | 목표 acc |
|---|---|---|---|
| keras36_cnn4_fashion | fashion_mnist (의류 흑백 28×28) | 10 | 0.92 |
| keras36_cnn5_cifar10 | cifar10 (컬러 32×32) | 10 | 0.67 |
| keras36_cnn6_cifar100 | cifar100 (컬러 32×32) | 100 | 0.4 |

- cifar는 컬러라 채널이 3이다(`(32,32,3)`). cifar100은 클래스가 100개로 많아 목표 acc가 낮게 설정된다.
- GPU 환경에서 실행한다.

---

## 3. CIFAR 데이터셋 수동 설치 (다운로드가 막힐 때)

`cifar10.load_data()`가 다운로드 서버(토론토대) 연결 실패(Timeout/404)로 막힐 때, 파일을 직접 받아 넣는다. (USB로 파일을 받아 넣는 방식으로 해결함.)

**설치 순서**
1. `cifar-10-python.tar.gz`(또는 cifar-100) 파일을 구해서 **`C:\Users\<사용자명>\.keras\datasets`** 폴더에 넣는다. (`<사용자명>`은 각자 윈도우 계정명. 교육장 노트북은 `Admin`.)
2. 그 자리에서 **압축을 푼다.** 풀면 `cifar-10-batches-py`(cifar100은 `cifar-100-python`) 폴더가 생긴다. 케라스는 이 폴더를 찾으면 다운로드를 건너뛴다.
3. VSCode에서 **가상환경을 GPU(tf29x-gpu)로** 잡는다.
4. 코드에서 `import` 후 `load_data()`를 실행한다.

**확인·문제 해결**
- 실행했을 때 **다운로드가 시작되면** 파일 위치·이름이 잘못된 것이다(케라스가 폴더를 못 찾은 것).
- 그래도 안 되면 **VSCode를 껐다 켜서** 다시 실행한다.
- (참고: `.keras\datasets`에 올바른 이름의 `.tar.gz`만 넣어도 케라스가 자동으로 압축을 풀지만, 위처럼 미리 풀어 폴더를 만들어두면 확실하다.)

---

## 4. keras37 — input_shape는 (높이, 너비, 채널)

```python
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))
```
- 이미지 입력의 `input_shape=(5,5,1)`은 **(높이, 너비, 채널)** 을 뜻한다(샘플 수는 제외). 채널 1은 흑백, 3은 컬러다. 이 점만 확인하기 위한 파일이다.

---

## 5. keras38 — padding과 stride (이미지 크기 조절)

### 문제 — Conv2D는 이미지를 계속 줄인다
커널을 적용할 때마다 이미지가 작아진다(예: `28×28×10` → 커널 (3,3) → `26×26×10`). 이미지 **외곽에 중요한 정보**가 있을 수 있는데, 계속 줄어들면 그 정보가 손실된다.

### padding — 외곽을 0으로 채워 크기 유지
```python
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), padding='same'))
```
- **`padding='same'`**: 입력 외곽을 0으로 둘러싸서, 출력 크기를 **입력과 같게** 유지한다.
- **`padding='valid'`**(기본값): 패딩을 적용하지 않아 크기가 줄어든다.

### stride — 커널의 이동 보폭
```python
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), strides=2, padding='same'))
```
- **`strides`**: 커널이 한 번에 몇 칸씩 이동하는지(보폭). **기본값은 1.**
- `strides=2`면 두 칸씩 건너뛰며 훑으므로, 출력 크기가 대략 절반으로 작아진다. (padding='same' + strides=2면 출력 ≈ 입력 ÷ 2.)
- strides를 2 이상 사용하는 경우 정확도는 떨어질 수 있다. 다만, 계산량이 줄어 속도가 빨라지기 때문에 '정확도 손해와 속도 이득의 트레이드 오프' 정도로 보고 있다.
- strides를 사용해 미리 크기를 줄여두면 Flatten 했을 때 더욱 부담이 줄어든다.
- 세밀한 것까지 다 보면 오히려 과적합이 발생할 수 있는데, 적당히 건너뛰면 큰 특징 위주로 봐서 일반화에 도움되는 경우도 있다.

---

## 6. keras39 — MaxPooling (크기를 반으로, 연산 없이)

```python
from tensorflow.keras.layers import MaxPooling2D
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), strides=1, padding='same'))
model.add(MaxPooling2D())   # 크기를 반으로 (예: 8×8 → 4×4)
```
- **`MaxPooling2D`**: 일정 영역(기본 2×2)에서 **가장 큰 값만 남겨** 이미지 크기를 반으로 줄인다. (예: 8×8 → 4×4)
- **학습 파라미터가 0이다.** 가중치를 학습하는 게 아니라 그냥 최댓값을 고르는 연산이라 **연산 비용이 거의 없다.**
- 크기를 줄여 계산량을 낮추면서도 강한 특징(최댓값)은 남기므로, 성능이 좋아지는 경우도 있어 효율적인 기능이다.
- (코드는 다음 시간에 이어서 작성 예정.)

---

## 7. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras36_cnn3_mnist | CNN 완성 | 4차원 reshape, Conv2D→Flatten→Dense→softmax |
| keras36_cnn4~6 | 이미지 실습 | fashion / cifar10 / cifar100 |
| (수동 설치) | 데이터셋 | .keras\datasets에 넣고 압축 풀기 |
| keras37 | input_shape | (높이, 너비, 채널) |
| keras38 | padding·stride | same(크기 유지)/valid, strides(보폭) |
| keras39 | MaxPooling | 크기 반으로, 파라미터 0 |

---

## 8. 오늘 한 줄 요약

> CNN은 이미지를 **4차원(샘플·높이·너비·채널)** 으로 만들어 **Conv2D로 특징을 뽑고**, **Flatten으로 펼친 뒤** Dense로 분류한다(→ softmax). Conv2D를 지나면 이미지가 작아지므로, **padding='same'** 으로 크기를 유지하거나 **strides**로 보폭을 조절하고, **MaxPooling**으로 파라미터 없이 크기를 반으로 줄인다.
> 내장 데이터 다운로드가 막히면 파일을 `.keras\datasets`에 직접 넣고 압축을 풀어 해결한다.
