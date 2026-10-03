# 2026.10.02 복습 — 모델별 차원 총정리 · 텍스트 분류(임베딩) · LSTM 재구성 · Reshape 층

> 오늘 흐름: 지금까지 배운 DNN·CNN·RNN·Embedding의 **차원 규칙을 한 표로 총정리** → reuters·imdb 텍스트 분류 → `sparse_categorical_crossentropy`(원핫 자동) → 여러 데이터를 **LSTM으로 재구성**(reshape 규칙) → 모델 안에서 차원을 바꾸는 **Reshape 층**. 전체 주제: **"모델마다 요구하는 차원과, 그 사이를 잇는 변환"**.

---

## 1. ★ 모델별 차원 총정리 (핵심 이론)

| 구분 | DNN | CNN | RNN | Embedding |
|---|---|---|---|---|
| **x(입력) 차원** | 2 | 4 | 3 | 2 |
| **input_shape 숫자 개수** | 1 | 3 | 2 | 1 |
| **output 차원** | 2 | 4 | 2 | 3 |
| **return_sequences** | — | — | 3 | — |
| **Flatten 후** | 2 | 2 | 2 | 2 |

### 한 줄씩 해석
- **입력 차원**: DNN `(N, 피처)` 2D / CNN `(N, H, W, C)` 4D / RNN `(N, 타임스텝, 피처)` 3D / Embedding `(N, 길이)` 2D(정수 시퀀스).
- **input_shape 숫자 개수**(맨 앞 N 제외): DNN 1개(`input_dim`) / CNN 3개`(H,W,C)` / RNN 2개`(타임스텝,피처)` / Embedding 1개(`input_length`).
- **출력 차원**:
  - DNN → 2D `(N, units)`
  - CNN(Conv2D) → 4D `(N, H', W', 필터)`
  - RNN → 2D `(N, units)` (마지막 타임스텝만 내보냄)
  - **Embedding → 3D** `(N, 길이, output_dim)` ← 2D를 넣으면 3D가 나옴(각 단어가 벡터가 되므로)
- **return_sequences**: **RNN에만** 있는 옵션. `True`면 모든 타임스텝을 내보내 출력이 **3D**로 유지됨(RNN 여러 층 쌓을 때 필요). 다른 층엔 없음.
- **Flatten**: 어떤 차원이 들어와도 **2D `(N, 전체)`** 로 펼침.

### 이 표가 설명하는 "연결 규칙"
- **Embedding + RNN이 자연스러운 이유**: Embedding(2D→**3D**) → RNN(3D 입력 OK → 2D) → Dense. 중간 reshape 불필요.
- **CNN은 Flatten이 필요한 이유**: Conv2D(4D 출력) → Dense(2D 필요)로 바로 못 감 → **Flatten(4D→2D)** 거쳐야 함.
- **CNN → RNN은 Reshape 필요**: Conv2D(4D) → LSTM(3D 필요)이라 차원이 안 맞음 → **Reshape(4D→3D)** 로 변환(오늘 Reshape2에서 실습).

---

## 2. 텍스트 분류 — 임베딩 실습 (keras62)

어제 배운 Embedding을 실제 데이터셋에 적용. 둘의 **클래스 수가 달라** 설정이 바뀐다.

### keras62_1_reuters — 다중분류 (46개 카테고리)
```python
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 핵심 전처리
pad_x_train = pad_sequences(x_train, padding='pre', maxlen=maxlen, truncating='post')
pad_x_test  = pad_sequences(x_test,  padding='pre', maxlen=maxlen, truncating='post')  # 같은 maxlen
y_train = ohe.fit_transform(y_train.reshape(-1,1))   # 원핫 (N, 46)

model.add(Embedding(input_dim=num_words, output_dim=100, input_length=maxlen))
model.add(LSTM(200))
...
model.add(Dense(46, activation='softmax'))   # 46개 카테고리
# loss='categorical_crossentropy'
```
- **46개 라벨 → 다중분류**: 출력 `Dense(46, softmax)`, loss `categorical_crossentropy`, y 원핫 필요.
- 실행 결과: **acc 약 0.75**.
- 자주 났던 실수: fit에 패딩 안 한 `x_train`을 넣음(ragged라 텐서 변환 실패), `maxlen`을 패딩보다 늦게 선언(불일치), test maxlen에 샘플 개수(2246)를 넣음. **maxlen은 맨 위에 변수로 한 번만 선언해 세 곳(train·test 패딩, input_length)에서 공유**가 안전.

### keras62_2_imdb — 이진분류 (긍정/부정)
```python
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)  # test_split 없음!
...
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=maxlen))
model.add(GRU(200))         # LSTM 대신 GRU도 사용
...
model.add(Dense(1, activation='sigmoid'))   # 이진분류
# loss='binary_crossentropy'
```
- **imdb는 `test_split` 옵션이 없다**(이미 train/test 25000씩 나뉘어 제공). reuters 전용 인자를 복사하면 에러.
- **2개 라벨 → 이진분류**: 출력 `Dense(1, sigmoid)`, loss `binary_crossentropy`, y 원핫 불필요(이미 0/1).
- 실행 결과: **LSTM acc 약 0.854, GRU acc 약 0.863**.

| | reuters | imdb |
|---|---|---|
| 클래스 | 46개(다중) | 2개(이진) |
| 데이터 분리 | `test_split=0.2` 사용 | 이미 분리됨(옵션 없음) |
| y 원핫 | 필요 | 불필요 |
| 출력층 | `Dense(46, softmax)` | `Dense(1, sigmoid)` |
| loss | categorical_crossentropy | binary_crossentropy |

---

## 3. sparse_categorical_crossentropy — 원핫 자동 (keras63)

다중분류에서 **y를 원핫하지 않고 정수 그대로** 쓰게 해주는 loss다.

```python
# y 원핫 작업 없음 (정수 라벨 그대로)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])

# 예측
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   # 예측은 argmax 필요 (확률 → 라벨)
# y_test = np.argmax(y_test, axis=1)       # ★ 삭제! y_test는 이미 정수
```

### categorical vs sparse 비교
| | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
|---|---|---|
| y 형태 | 원핫 `(N, 클래스)` | **정수 `(N,)`** |
| y 원핫 작업 | 필요 | **불필요**(내부 자동) |
| `y_predict` argmax | 필요 | 필요 |
| **`y_test` argmax** | 필요 | **불필요** ← 차이 |

- 흔한 실수: 원핫 버전 코드를 복사하면 `y_test = np.argmax(y_test, axis=1)`가 딸려온다. sparse에선 y_test가 1차원 정수라 argmax가 "axis 1 out of bounds" 에러를 낸다 → **그 줄 삭제**.
- 실행 결과(mnist·fashion·cifar10·cifar100, CNN): acc **0.990 / 0.904 / 0.673 / 0.252**. (cifar100은 100클래스라 난도가 높아 낮음.)

---

## 4. DNN·CNN → LSTM 재구성 (keras64)

표·이미지 데이터를 **LSTM(3D 입력)에 맞게 reshape**해서 돌려보는 실습. 핵심은 **"스케일링(2D) → reshape(3D)" 순서**와 **데이터 전체 shape의 맨 앞(샘플 수)은 input_shape에서 제외**.

### 공통 변환 규칙
```python
# 표 데이터 (예: california 피처 8개)
x = x.reshape(-1, 8, 1)              # (N, 8) → (N, 8, 1) : 피처를 타임스텝으로
model.add(LSTM(40, input_shape=(8, 1)))

# 이미지 (예: mnist 28×28)
x_train = x_train.reshape(-1, 28*28, 1)   # (N,28,28) → (N,784,1) : 픽셀을 타임스텝으로
model.add(LSTM(40, input_shape=(28*28, 1)))
```

### 자주 났던 실수 (순서·차원)
- **스케일러는 2D만 받는다**: `x.reshape(-1,30,1)`(3D)을 먼저 하고 스케일링하면 "dim 3 … dim ≤ 2 required" 에러. **순서는 split → 스케일링(2D) → reshape(3D)**.
- **증폭(datagen.flow)은 4D 이미지 필요**: LSTM용 `(N,784,1)`로 먼저 바꾸면 증폭이 안 됨. **증폭·합치기는 이미지(4D)로 끝낸 뒤 맨 마지막에 `(N,784,1)`로 변환**.

### 실행 결과 — "데이터에 맞는 모델"의 증거 (정직하게)
| 데이터 | 형태 | LSTM 결과 |
|---|---|---|
| california (회귀, 피처 8) | (N, 8, 1) | loss 약 0.37~0.54 |
| cancer (이진, 피처 30) | (N, 30, 1) | acc 약 0.98 |
| digits (8×8 작은 이미지) | (N, 64, 1) | acc 약 0.97 |
| **mnist (28×28)** | (N, 784, 1) | **acc 약 0.11** ← 사실상 랜덤(실패) |
| fashion (28×28) | (N, 784, 1) | acc 약 0.75 |
| cifar10 (32×32×3) | (N, 1024, 3) | acc 약 0.66 |

- **mnist가 LSTM에서 acc 0.11(10개 중 찍기 수준)로 사실상 실패**한 것이 핵심 교훈이다. 784개 타임스텝은 너무 길어 LSTM이 앞을 잊고, 애초에 이미지는 공간 구조(CNN)가 맞는 데이터라 순차 모델엔 부적합하다.
- 작은 이미지(digits 8×8=64스텝)나 표 데이터는 그럭저럭 나오지만, **큰 이미지일수록 LSTM은 불리**하다. "**이미지 → CNN, 시계열 → RNN**"을 수치로 체감하는 실습.
- LSTM은 타임스텝을 순차 처리해 **CNN보다 훨씬 느리다**(이미지 기준 수백 초).

---

## 5. Reshape 층 — 모델 안에서 차원 바꾸기 (keras65)

지금까지 reshape는 **데이터 단계**(`x.reshape(...)`)에서 했지만, **모델 안에 `Reshape` 층**을 넣어 층과 층 사이에서 차원을 바꿀 수도 있다.

### keras65_Reshape1 — Dense·Reshape로 이미지 만들어 CNN에 투입
```python
from tensorflow.keras.layers import Reshape

model.add(Dense(280, input_shape=(28, 28)))      # (N,28,28) → (N,28,280)
model.add(Reshape(target_shape=(28, 28, 10)))    # (N,28,280) → (N,28,28,10)  4D로
model.add(Conv2D(64, (3,3)))                     # 이제 CNN 투입 가능
...
```
- 데이터를 4D로 안 바꾸고 들어와도, **모델 안에서 `Reshape`로 4D를 만들어** Conv2D에 넣는다.
- `Reshape`는 **총 원소 수가 같아야** 한다: 28×280 = 28×28×10 = 7840 ✓.
- 실행 결과 acc 약 0.986.

### keras65_Reshape2 — CNN → Reshape → LSTM 연결
```python
model.add(Reshape(target_shape=(28,28,1), input_shape=(28,28)))  # 처음부터 4D
model.add(Conv2D(64, (3,3)))            # (26,26,64)
model.add(Conv2D(32, (3,3), ...))       # (24,24,32)
model.add(Conv2D(32, (2,2), ...))       # (23,23,32)
model.add(Reshape(target_shape=(23*23, 32)))   # ★ 4D → 3D : (23,23,32)→(529,32)
model.add(LSTM(10))                      # LSTM(3D) 투입
```
- **CNN(4D) → LSTM(3D) 연결의 핵심**: `Reshape(H×W, 채널)` = **공간을 타임스텝으로, 채널(필터 수)을 피처로** 펼친다. `(23,23,32)` → `(529, 32)`.
- **채널 숫자는 마지막 Conv의 필터 수**(32)를 그대로 써야 한다. 흔한 실수로 16 등 다른 수를 넣으면 "total size … must be unchanged" 에러(23×23×32 ≠ 529×16).

### Reshape의 철칙
```
Reshape 전 원소 총수 == Reshape 후 원소 총수
```
- 모양만 바꾸고 값 개수는 안 변한다. 헷갈리면 `model.summary()`로 직전 층의 Output Shape를 보고 그 숫자들로 맞춘다.

---

## 6. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras62_1_reuters | 텍스트 다중분류 | 46클래스, softmax, categorical, maxlen 공유 (acc~0.75) |
| keras62_2_imdb | 텍스트 이진분류 | test_split 없음, sigmoid, binary, GRU도 사용 (acc~0.86) |
| keras63_sparse1~4 | 원핫 자동 | sparse_categorical_crossentropy, y_test argmax 삭제 |
| keras64_LSTM01~06 | LSTM 재구성 | 스케일링(2D)→reshape(3D), mnist는 acc 0.11로 실패(이미지=CNN) |
| keras65_Reshape1 | 모델 내 Reshape | Dense→Reshape→Conv2D, 총원소 수 보존 |
| keras65_Reshape2 | CNN→LSTM 연결 | Reshape(H×W, 채널)로 4D→3D |

---

## 7. 오늘 한 줄 요약

> 모델마다 입력 차원이 정해져 있다: **DNN 2D · RNN 3D · CNN 4D · Embedding은 2D 입력→3D 출력**. input_shape엔 맨 앞 샘플 수를 빼고 넣으며, RNN의 `return_sequences=True`는 출력을 3D로 유지(층 쌓기용), Flatten은 무엇이든 2D로 편다.
> 텍스트 분류는 클래스 수로 갈린다 — **reuters(46, softmax/categorical)** vs **imdb(2, sigmoid/binary, test_split 없음)**. 다중분류는 **`sparse_categorical_crossentropy`** 로 y 원핫을 생략할 수 있고(그 경우 `y_test` argmax도 삭제).
> 데이터를 LSTM으로 바꿀 땐 **스케일링(2D)→reshape(3D)** 순서를 지키고, 증폭은 이미지(4D)로 끝낸 뒤 변환한다. 다만 **큰 이미지는 LSTM에서 성능이 급락**(mnist acc≈0.11)해 "이미지=CNN"을 확인시킨다.
> 모델 안에서 차원을 바꾸는 **`Reshape` 층**은 총 원소 수를 보존해야 하며, **CNN→LSTM 연결은 `Reshape(H×W, 채널)`** 로 공간을 타임스텝, 채널을 피처로 펼친다.
