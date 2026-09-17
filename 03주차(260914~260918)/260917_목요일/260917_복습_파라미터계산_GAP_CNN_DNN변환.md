# 2026.09.17 복습 — Dropout·MaxPooling 순서 · Conv2D 파라미터 계산 · GlobalAveragePooling · CNN↔DNN 변환

> 오늘 흐름: MaxPooling·Dropout 순서 비교 → Conv2D의 파라미터가 어떻게 계산되는지 → Flatten의 파라미터 폭증 문제를 푸는 GlobalAveragePooling → CNN을 DNN으로, DNN을 CNN으로 서로 변환해 성능 비교.

---

## 1. keras39_MaxPooling1~4 — Dropout과 MaxPooling의 순서

maxpooling·stride·padding을 적용해 mnist·fashion·cifar10·cifar100 성능을 올리는 실습이다. 이때 **Dropout과 MaxPooling의 순서**를 비교한다.

- **Conv2D → Dropout → MaxPooling**: 노드 일부를 제외하고 계산한 뒤, 그 결과를 반으로 줄인다.
- **Conv2D → MaxPooling → Dropout**: 먼저 반으로 줄인 뒤, 노드 일부를 제외한다.

- 통상적으로는 **Conv2D → Dropout → MaxPooling** 순서를 쓴다. 다만 실습에서 반대 순서(**Conv2D → MaxPooling → Dropout**)로 더 좋은 결과를 낸 경우도 있었다.
- 어느 순서가 나은지는 데이터·모델마다 다르므로 정해진 답이 없고, 돌려보고 판단한다. (이번 실습에서는 네가지 데이터셋 모두 성능 향상이 있었다.)

---

## 2. Conv2D 파라미터 계산 (4차원)

Dense의 파라미터가 `(입력 × 출력) + 출력`이었듯, Conv2D도 파라미터(가중치 + bias)를 계산할 수 있다.

### 공식
$$ \text{파라미터} = \text{필터 수} \times (\text{커널 높이} \times \text{커널 너비} \times \text{입력 채널 수} + 1) $$

- 괄호 안 `커널 × 커널 × 입력 채널` = 가중치 수, `+1` = bias.
- **입력 채널 수 = 바로 앞 층의 필터 수** (첫 층은 이미지 채널: 흑백 1, 컬러 3).

### 계산 예시 (입력 (5,5,1) 기준)
```python
model.add(Conv2D(4, (2,2), input_shape=(5,5,1)))   # 1층: 필터 4, 커널 2×2, 입력 채널 1
model.add(Conv2D(3, (2,2)))                         # 2층: 필터 3, 커널 2×2, 입력 채널 4
```

| 층 | 필터 | 커널 | 입력 채널 | 계산 | 파라미터 |
|---|---|---|---|---|---|
| 1층 | 4 | 2×2 | 1 | 4 × (2×2×1 + 1) = 4×5 | **20** |
| 2층 | 3 | 2×2 | **4** | 3 × (2×2×4 + 1) = 3×17 | **51** |

- 2층의 입력 채널이 **4**인 것은 앞 층의 필터 수(4)가 그대로 다음 층의 입력 채널이 되기 때문이다.
- Conv2D 한 칸의 연산도 결국 `wx + b`(커널이 w) 형태다.

---

## 3. GlobalAveragePooling(GAP) — Flatten의 파라미터 폭증 해결

### Flatten의 단점
Conv2D 뒤에 **Flatten**을 쓰면 공간 전체를 펼쳐 Dense와 연결한다. 그런데 이때 **파라미터가 너무 많이 생긴다.** 예를 들어 `(N, 20, 20, 16)`을 Flatten하면 `20×20×16 = 6400`이 되고, 이 6400개가 다음 Dense와 전부 연결되어 파라미터가 폭증한다. → **과적합**이 생기고 속도·성능에 부담이 된다.

### GAP — 채널마다 평균 하나로 압축
```python
from tensorflow.keras.layers import GlobalAveragePooling2D
# model.add(Flatten())               # 대신
model.add(GlobalAveragePooling2D())  # 이걸 사용
```
- **GlobalAveragePooling2D**: 각 채널(필터)의 공간값 전체를 **평균 하나**로 압축한다. `(N, 높이, 너비, 채널)` → `(N, 채널)`.
- 예: `(N, 20, 20, 16)` → `(N, 16)`. Flatten이 6400개를 만드는 자리에서 GAP은 16개만 만든다.
- 값의 개수가 급감하므로 뒤 Dense의 파라미터가 크게 줄어든다. → **과적합이 줄고 속도가 빨라지며, 성능이 오르는 경우가 많다.**

---

## 4. keras41_dnn1~4 — CNN 모델을 DNN으로 변환

이미지를 Conv2D(CNN)가 아니라 Dense(DNN)로 학습하도록 바꾼다.
```python
x_train = x_train.reshape(-1, 28 * 28 * 1)   # (60000, 28, 28) → (60000, 784) : 2차원(DNN)으로 펼침
...
model.add(Dense(64, input_shape=(784,), activation='relu'))  # Conv2D 없이 Dense만
...
model.add(Dense(10, activation='softmax'))
```
- 이미지를 처음부터 1차원으로 펼쳐(`28×28=784`) Dense에 바로 넣는다. Conv2D·Flatten이 없다.
- **통상 이미지 훈련에서는 DNN이 CNN을 이기지 못한다.** CNN은 이미지의 공간 구조(옆 픽셀 관계)를 활용하는데, DNN은 픽셀을 그냥 나열된 숫자로 보기 때문이다.
- 다만 단순한 이미지 등에서는 예외적으로 DNN이 비슷하거나 나은 경우도 있어, 직접 해보고 확인한다.

---

## 5. keras42_cnn1~10 — DNN(표 데이터)을 CNN으로 변환

반대로, 표(정형) 데이터를 4차원으로 만들어 Conv2D에 넣어본다.
```python
x_train = x_train.reshape(-1, 200, 1, 1)             # 표 데이터를 4차원으로 (예: 산탄데르 피처 200개)
model.add(Conv2D(64, (2,1), input_shape=(200,1,1), padding='same'))
```
- 표 데이터는 **공간 구조가 없다**(컬럼 순서를 바꿔도 의미가 같음). 그래서 Conv2D에 넣어도 CNN의 장점(공간 패턴 추출)이 살지 않아 성능이 특별히 좋아질 이유가 없다.
- 실제로 대부분 나아지지 않았고, 1~2개 데이터셋만 조금 좋아졌다. 이 변환은 실무 방식이라기보다 "형태만 맞추면 넣을 수는 있다"를 확인하는 연습이다.
- (표 데이터를 reshape할 때는 그 데이터의 **피처 개수**를 정확히 써야 한다. 예: 산탄데르는 200개이므로 `reshape(-1, 200, 1, 1)`, `input_shape=(200,1,1)`.)

---

## 6. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras39_MaxPooling1~4 | Dropout·MaxPooling 순서 | 통상 Conv→Dropout→MaxPool, 반대도 가능(실험) |
| (파라미터 계산) | Conv2D 파라미터 | 필터수 × (커널×커널×입력채널 + 1) |
| keras40_GAP01~04 | GlobalAveragePooling | Flatten 대신 채널 평균으로 압축, 파라미터 급감 |
| keras41_dnn1~4 | CNN→DNN 변환 | 이미지를 784로 펼쳐 Dense만 사용 |
| keras42_cnn1~10 | DNN→CNN 변환 | 표 데이터를 4차원으로, 대부분 이득 없음 |

---

## 7. 오늘 한 줄 요약

> Conv2D의 파라미터는 **필터 수 × (커널 × 커널 × 입력 채널 + 1)** 로 계산되며, 입력 채널은 앞 층의 필터 수다. Flatten은 공간을 펼치며 파라미터가 폭증해 과적합을 부르므로, **GlobalAveragePooling**으로 채널마다 평균 하나로 압축하면 파라미터가 급감하고 속도·성능이 개선된다.
> 이미지는 보통 CNN이 DNN보다 낫고(공간 구조 활용), 표 데이터를 CNN으로 바꾸는 것은 대체로 이득이 없다. 둘 다 예외가 있어 직접 돌려 확인한다.
