# 2026.09.10 복습 — softmax 심화 · summary(파라미터) · input_shape · 스케일링(MinMaxScaler)

> 오늘 흐름: softmax를 수식으로 더 파고들고 → `model.summary()`로 파라미터 개수 이해 → `input_shape` 개념 → 데이터 스케일링(MinMaxScaler)과 **스케일러를 train 기준으로 맞춰야 하는 이유**.

---

## 1. softmax 심화

- softmax를 통과한 출력들을 **모두 더하면 정확히 1.0** 이 되며, 1.0을 초과할 수 없다. 각 출력은 "그 클래스일 확률"이 된다.

### 수식
강의에서 든 예시는 예측값 1, 2, 7을 각각 합(10)으로 나누는 형태다.
```
1/(1+2+7) = 0.1
2/(1+2+7) = 0.2
7/(1+2+7) = 0.7   → 합 = 1.0
```
- 이것은 "출력을 합이 1이 되게 만든다"는 개념을 단순화한 설명이다.
- **실제 softmax**는 각 값에 자연상수 e의 지수(**e^x**)를 먼저 취한 뒤 나눈다.

$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$

- 지수(e^x)를 쓰는 이유: ① e^x는 항상 양수라 음수 입력도 양수로 바꿔 음수를 막는다. ② 큰 값과 작은 값의 차이를 더 벌려 확률 분포를 뚜렷하게 만든다.

### loss는 argmax 이전에 계산된다
```
softmax 출력(ŷ) : [0.1, 0.2, 0.7]   ← 이 확률값으로 loss 계산
실제값(y, 원-핫): [0,   1,   0  ]
```
- **loss(categorical_crossentropy)는 softmax 확률값으로 계산**한다(argmax 이전). 위 예시는 ŷ의 최댓값 위치(2)와 정답 위치(1)가 달라 예측이 틀린 경우다.
- **argmax는 최종 예측 클래스를 뽑을 때만** 사용하며, loss 계산에는 관여하지 않는다.

---

## 2. keras24_kaggle_santander — 이진분류를 다중분류로 재구성

이진분류는 2개 중 하나를, 다중분류는 여러 개(n개) 중 하나를 찾는다. **2개도 "여러 개"에 포함되므로, 이진분류는 다중분류의 특수한 경우로 볼 수 있다.**

그래서 이진분류였던 산탄데르를 다중분류 방식으로 재구성할 수 있다.

| 구분 | 이진분류 방식 | 다중분류 방식(2 클래스) |
|---|---|---|
| 원-핫 | 하지 않음 | get_dummies로 (n, 2) |
| 출력층 | `Dense(1, sigmoid)` | `Dense(2, softmax)` |
| loss | binary_crossentropy | categorical_crossentropy |
| 예측 후처리 | np.round | np.argmax |

- 두 방식 모두 유효하다. 다만 이렇게 이진분류를 굳이 다중분류로 푸는 경우는 드물며, 이런 구성이 필요한 상황도 있어 실습해 본 것이다.

---

## 3. keras25_summary — model.summary()와 파라미터 계산

```python
model.summary()   # 층별 출력 shape와 파라미터 개수를 표로 출력
```

**파라미터 = 학습으로 값이 바뀌는 개별 숫자.** `y = wx + b`에서 w(가중치)와 b(편향)는 학습으로 바뀌므로 파라미터이고, x(입력)는 고정이라 파라미터가 아니다.

- 한 Dense 층의 파라미터 수 = **(입력 수 × 출력 노드 수) + 출력 노드 수**
  - 앞부분 `입력 × 출력` = 가중치 w의 개수(모든 연결마다 w 하나).
  - 뒷부분 `+ 출력` = 편향 b의 개수(노드마다 b 하나). **summary는 이 bias를 포함해 계산한다.**

예시 모델(`Dense(3, input_dim=1)` → `Dense(4)` → `Dense(3)` → `Dense(1)`):

| 층 | 입력→출력 | w 개수 | b 개수 | 파라미터 |
|---|---|---|---|---|
| dense | 1→3 | 3 | 3 | 6 |
| dense_1 | 3→4 | 12 | 4 | 16 |
| dense_2 | 4→3 | 12 | 3 | 15 |
| dense_3 | 3→1 | 3 | 1 | 4 |

- 합계 = **41** (Total params와 일치).
- 참고: 스칼라 직관으로는 `y = wx + b`지만, 실제 행렬 계산에서는 데이터 X `(샘플 수, 피처 수)`와 가중치 W `(피처 수, 노드 수)`를 곱해야 차원이 맞으므로 **`y = XW + b`(X를 앞에)** 형태가 된다. 행렬 곱은 앞 행렬의 열 수와 뒤 행렬의 행 수가 같아야 성립하기 때문이다.

---

## 4. keras26_input_shape — input_dim과 input_shape

```python
# 아래 두 줄은 같은 의미
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, input_shape=(4,), activation='relu'))
```
- **`input_dim=4`와 `input_shape=(4,)`는 같다.**
- **`input_shape`는 원본 데이터 shape에서 첫 차원(샘플 수 n)을 뺀 나머지**를 튜플로 적는다.

| 원본 데이터 shape | input_shape |
|---|---|
| (n, 4) | (4,) |
| (n, 100, 3) | (100, 3) |
| (n, 100, 100, 3) | (100, 100, 3) |

- `input_dim`은 1차원 입력에만 쓸 수 있고, `input_shape`는 다차원 입력(이미지 등)에도 쓸 수 있어 더 일반적이다.

---

## 5. 스케일링(MinMaxScaler) — 데이터를 0~1로 줄이기 ★

### 왜 스케일링을 하나
- 데이터 값이 너무 크거나 컬럼마다 범위가 제각각이면 학습에 불리할 수 있다. 그래서 **모든 데이터를 같은 비율로 줄여** 비슷한 범위(0~1)로 맞춘다.

### MinMaxScaler 공식
각 컬럼을 그 컬럼의 최솟값~최댓값 기준으로 0~1로 변환한다.

$$ \text{변환값} = \frac{\text{원값} - \text{Min}}{\text{Max} - \text{Min}} $$

- 원값이 그 컬럼의 최솟값이면 0, 최댓값이면 1이 된다.
- 이 공식은 원본에 음수가 있어도 정상 동작한다(최솟값이 0으로 매핑되므로). 사이킷런이 이 기능을 제공한다.
- 변환 후 최댓값이 `1.0000...2`처럼 나오는 것은 파이썬 부동소수점 연산 오차이며 1로 보면 된다.

### fit과 transform의 차이
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)              # ① 학습: x_train 각 컬럼의 Min·Max를 계산해 저장 (데이터는 안 바꿈) // train으로 변환 규칙(min·max)을 정함
x_train = scaler.transform(x_train)   # ② 변환: 저장된 Min·Max로 실제 변환  // train 변환 → 값이 0~1로 줄어듦
x_test  = scaler.transform(x_test)    # ③ 변환: 같은(train의) Min·Max로 test도 변환  // test도 같은 규칙으로 변환 → 0~1로 줄어듦
```
- **fit**: 변환에 필요한 기준값(MinMax의 경우 각 컬럼의 min·max)을 **계산해서 저장만** 한다. 데이터 자체는 바꾸지 않는다.
- **transform**: fit으로 저장한 기준값을 이용해 **실제로 데이터를 변환**한다.
- test에는 **fit하지 않고 transform만** 한다. test에 다시 fit하면 test의 min·max를 쓰게 되어 train과 기준이 달라진다.

### x_test 를 transform 하지 않으면? 
- 모델은 x_train으로 학습했으니 입력은 0~1범위라고 배웠음 
- 평가할 때 x_test로 100, 5000과 같은 원본 값을 던지면 모델이 한번도 본적 없는 큰 값들이라 예측이 엉망이 됨

### 비유
- 키를 cm로 재서 학습시킨 모델이 있음
- 그럼 예측할 때도 cm로 넣어야 함, mm로 넣으면 170cm가 1700mm로 됨
- 모델이 "이 사람은 키가 1700이네?" 라고 잘못 판단함

### 비유를 통해 외우는 방법
- scaler.fit(x_train) = cm라는 단위(변환 규칙)를 정한다
- transform(x_train) = train을 cm로 변환
- transform(x_test) = test도 같은 cm로 변환 ← 이걸 빼먹으면 test만 mm로 들어가서 엉망

### 왜 스케일러를 train 기준으로 맞추나 (오늘의 핵심)

**test는 "아직 보지 못한 미래 데이터"로 취급해야 한다.** 스케일러를 전체 데이터(train+test)에 fit하면, test의 min·max가 스케일러에 반영되어 **test 정보가 전처리 단계에서 새어 들어간다(정보 누수)**. 실제 배포 상황에서는 미래 데이터가 없으므로, 스케일러는 **train만으로** 만들어야 한다.

그 과정을 그림으로 보면:
```
전체 데이터 x            (원래 범위 0 ~ 1.0)
│
├── Train : scaler.fit(x_train) → transform(x_train)   → 0 ~ 1 로 변환됨
│
└── Test  : scaler.transform(x_test)                   → train 기준으로 변환
                                                          (0 밑이나 1 위로도 나올 수 있음)
```
- **fit은 Train으로만** 하고(기준 = train의 min·max), Train과 Test **둘 다 그 기준으로 transform** 한다.
- Test를 train 기준으로 변환하므로, test에 train보다 작거나 큰 값이 있으면 **변환값이 0 미만 또는 1 초과**로 나온다. 이는 오류가 아니라 **정상**이며, 오히려 "test를 train 기준으로 처리했다"는 증거다. (새 데이터가 train 범위를 벗어날 수 있는 현실을 반영.)

### keras27(잘못된 방식) vs keras28(올바른 방식)
```python
# keras27 — 잘못된 방식
scaler.fit(x)              # 전체 데이터에 fit → test까지 포함됨 (정보 누수)
x = scaler.transform(x)
x_train, x_test, ... = train_test_split(x, y, ...)   # 그다음 분리

# keras28 — 올바른 방식
x_train, x_test, ... = train_test_split(x, y, ...)   # 먼저 분리
scaler.fit(x_train)                # train에만 fit
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test) # 같은 기준으로 test 변환
```
- keras27은 **분리 전 전체 데이터에 fit**해서 test 정보가 스케일러에 반영된다. 실무에서 쓰지 않는 잘못된 방식이다.
- keras28은 **분리 후 train에만 fit**하고 양쪽을 transform한다. 실제 keras28 출력에서 x_train은 0~1, x_test는 `-0.0003 ~ 1.016`으로 살짝 벗어나는데, 이것이 위에서 설명한 정상 동작이다.

---

## 6. keras28_scaler02 ~ 10 — 스케일 적용 효과 비교

같은 모델에 스케일 적용 전/후 결과를 여러 데이터셋에 비교한 실습이다.

| 결과 | 데이터셋 |
|---|---|
| 좋아짐 | boston, wine, covtype |
| 비슷함 | diabetes, kaggle bike, santander, digits |
| 안 좋아짐 | 따릉이, cancer |

- 스케일링이 항상 성능을 올리는 것은 아니며, 데이터에 따라 좋아지거나·비슷하거나·나빠질 수 있다.
- 이 실습은 시간 관계상 epochs를 적게 줬으므로 결과가 확정적이지 않다. 스케일 효과가 데이터 유형에 따라 어떻게 갈리는지는 더 검증이 필요하다.

---

## 7. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras24_kaggle_santander | 이진→다중 재구성 | 이진분류를 softmax(2 클래스)로 |
| keras25_summary | 파라미터 | (입력×출력)+출력, bias 포함 |
| keras26_input_shape | 입력 지정 | input_dim=4 ≡ input_shape=(4,), 첫 차원 제외 |
| keras27_scaler01 | 스케일링(잘못) | 전체에 fit → 정보 누수 |
| keras28_scaler01 | 스케일링(올바름) | 분리 후 train에만 fit, 양쪽 transform |
| keras28_scaler02~10 | 효과 비교 | 데이터별로 좋아짐/비슷함/나빠짐 |

---

## 8. 오늘 한 줄 요약

> 스케일링(MinMaxScaler)은 `(원값−Min)/(Max−Min)`으로 데이터를 0~1로 줄인다. **fit은 기준(min·max)을 계산·저장, transform은 그 기준으로 실제 변환**이며, 스케일러는 **train에만 fit**하고 train·test 모두 그 기준으로 transform한다.
> 이유는 test를 미지의 데이터로 취급해 정보 누수를 막기 위함이다. 그 결과 test 변환값이 0~1을 살짝 벗어나는 것은 정상이다. 스케일 효과는 데이터에 따라 다르다.
