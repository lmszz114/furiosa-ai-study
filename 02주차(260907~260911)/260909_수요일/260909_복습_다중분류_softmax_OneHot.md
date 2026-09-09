# 2026.09.09 복습 — 다중분류(softmax · categorical_crossentropy · OneHotEncoding · argmax)

> 오늘 흐름: 이진분류에 이어 **다중분류(범주 3개 이상)** 를 다룬다. 원-핫 인코딩으로 라벨을 변환하고, 출력층 softmax·손실 categorical_crossentropy로 모델을 구성하며, 예측은 argmax로 판정한다.

---

## 1. 회귀·분류 총정리

모든 모델은 y값의 종류에 따라 **회귀(수치 예측)** 또는 **분류(범주 예측)** 중 하나다. 회귀 → 이진분류 → 다중분류로 갈 때 달라지는 부분을 정리하면 다음과 같다.

| 구분 | 회귀 | 분류(이진) | 분류(다중) |
|---|---|---|---|
| 마지막 층 activation | linear(지정 안 함) | sigmoid | **softmax** |
| loss | mse, mae 등 | binary_crossentropy | **categorical_crossentropy** |
| predict 후처리 | 없음 | np.round | **np.argmax** |
| OneHot Encoding | X | X | **O** |
| 마지막 층 노드 수 | 출력 개수(N) | 1개 | **y의 클래스 개수** |

---

## 2. keras23_softmax1_onehot_iris — 다중분류 기본

### 데이터 (load_iris)
```python
from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                    # (150, 4) (150,)
print(np.unique(y, return_counts=True))    # (array([0,1,2]), array([50,50,50]))
```
- 붓꽃 데이터. 피처 4개, 클래스 3개(0, 1, 2)가 각 50개씩.

### 원-핫 인코딩(OneHot Encoding) — 왜 필요한가
라벨 0, 1, 2는 **크기나 순서가 아니라 서로 대등한 범주**다. 2가 1보다 두 배 크다거나 1+1=2 같은 관계가 아니다. 그런데 숫자 그대로 두면 모델이 2를 1보다 "큰 값"으로 오해할 수 있다.

그래서 각 범주를 **값의 크기가 아닌 위치(자리)** 로 표현한다.
```
0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]
```
- 이 변환으로 y의 형태가 벡터 `(150,)`에서 행렬 `(150, 3)`으로 바뀐다.
- 즉 원-핫 인코딩은 "라벨을 위치 벡터로 바꾸는 것"이다. (어떤 범주가 가장 잘 맞는지 고르는 것은 뒤의 softmax·argmax가 한다.)

### 원-핫 인코딩의 3가지 방법
**방법 1 — TensorFlow: to_categorical**
```python
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)    # (150,) → (150, 3)
```
- 주의: to_categorical은 **0번 컬럼부터 시작**하는 것이 기본이다. 라벨이 1부터 시작하는 데이터면(예: covtype는 1~7) 쓰지 않는 0번 컬럼을 강제로 만들어 클래스 수보다 열이 하나 많아진다.

**방법 2 — pandas: get_dummies**
```python
y = pd.get_dummies(y)            # 결과가 True/False로 표시됨
y = pd.get_dummies(y, dtype=int) # dtype=int를 주면 0/1로 표시됨
```

**방법 3 — sklearn: OneHotEncoder (reshape 필요)**
```python
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1, 1)                     # (150,) → (150, 1) : 2차원으로
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)
```
- OneHotEncoder는 **입력이 2차원 배열**이어야 한다. y가 벡터 `(150,)`이면 "2D array를 기대했는데 1D가 왔다"는 에러가 나므로, reshape로 `(150, 1)`로 만든 뒤 `fit_transform`에 넣는다.
- **reshape의 `-1`** 은 "그 자리 크기를 자동으로 계산하라"는 뜻이다. `reshape(-1, 1)`은 "열은 1개, 행 수는 알아서"이므로 `(150, 1)`이 된다.
- `sparse_output=False`를 주지 않으면 결과가 **희소 행렬(sparse matrix, 0이 많은 데이터를 효율적으로 저장하는 형식)** 로 나온다. 일반 배열로 받으려면 False를 지정한다.

### 원-핫을 train_test_split보다 먼저 하는 이유
```python
y = ...(원-핫 인코딩)...
x_train, x_test, y_train, y_test = train_test_split(x, y, ..., stratify=y)
```
- 원-핫을 **먼저 하면 인코딩을 한 번만** 하면 된다. 반대로 나눈 뒤에 하면 y_train과 y_test를 **각각(두 번)** 인코딩해야 한다.
- 또한 나눈 뒤 따로 인코딩하면, 한쪽 split에 없는 클래스 때문에 원-핫 **열 구성이 train과 test에서 어긋날 수 있다.** 전체 y를 먼저 인코딩하면 모든 클래스가 포함되어 열 구성이 일치한다.

### 모델 구성 — 출력층 softmax
```python
model.add(Dense(64, input_dim=4, activation='relu'))
...
model.add(Dense(3, activation='softmax'))   # 클래스 3개 → 노드 3개
```
- **마지막 층 노드 수 = 클래스 개수.** iris는 3개이므로 `Dense(3)`.
- **softmax**: 출력들을 **합이 정확히 1이 되는 확률 분포**로 만든다. 각 출력은 "그 클래스일 확률"이 된다(예: `[0.02, 0.13, 0.85]`). 여러 출력이 동시에 1이 되는 일 없이, 전체 합이 1이 되도록 조정한다.

### 컴파일 — categorical_crossentropy
```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
- **`categorical_crossentropy`**: 다중분류 전용 손실 함수. (이진분류의 binary_crossentropy에 대응.)
- metrics에 정확도(acc)를 넣는 것은 이진분류와 동일하다.

### 평가·예측 — argmax
```python
result = model.evaluate(x_test, y_test)
print('loss = ', result[0]); print('acc = ', round(result[1], 2))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   # 확률 중 가장 큰 위치 → 0/1/2
y_test    = np.argmax(y_test, axis=1)      # 원-핫도 위치로 되돌림 → 0/1/2

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_predict)
```
- softmax 출력은 확률 3개(연속값), y_test는 원-핫(위치 표시)이라 그대로 `accuracy_score`에 넣으면 형태가 맞지 않아 에러가 난다(mix of multilabel-indicator and continuous).
- **`np.argmax(..., axis=1)`**: 각 행에서 **가장 큰 값의 인덱스(위치)** 를 반환한다. 확률 `[0.02, 0.13, 0.85]` → `2`, 원-핫 `[0, 0, 1]` → `2`.
- **y_predict와 y_test 둘 다** argmax로 되돌려야 형태가 같아져 비교가 된다.
- 이진분류의 `np.round`에 대응하는 것이 다중분류의 `np.argmax`다.
- (변수명은 함수명 `accuracy_score`와 겹치지 않게 `acc_score` 등으로 두는 것이 안전하다.)

---

## 3. keras23_softmax2 ~ 4 — 실전 데이터 실습

데이터셋만 제시되고 나머지는 직접 구현하는 실습이다. 다중분류의 기본 형태(원-핫 → softmax → categorical_crossentropy → argmax)를 그대로 적용한다.

| 파일 | 데이터셋 | shape | 클래스 수 | 목표 acc |
|---|---|---|---|---|
| keras23_softmax2_wine | load_wine | (178, 13) | 3 (0~2) | 0.95 이상 |
| keras23_softmax3_fetch_covtype | fetch_covtype | (581012, 54) | 7 (**1~7**) | 0.93 이상 |
| keras23_softmax4_digits | load_digits | (1797, 64) | 10 (숫자 0~9) | 1.0 |

- 세 실습 모두 원-핫은 pandas `get_dummies`를 사용한다.
- **covtype 주의**: 라벨이 0이 아니라 **1부터 7까지**다. 이 경우 to_categorical을 쓰면 쓰지 않는 0번 열이 생겨 8열이 되지만, get_dummies는 실제 존재하는 라벨만 인코딩하므로 7열이 된다.
- digits는 8×8 손글씨 숫자 이미지가 64개 피처로 펼쳐진 데이터이며, 클래스는 0~9의 10개다(마지막 층 `Dense(10)`).

---

## 4. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras23_softmax1_onehot_iris | 다중분류 기본 | 원-핫 3가지 방법, softmax, categorical_crossentropy, argmax |
| keras23_softmax2_wine | 실습 | 클래스 3, get_dummies |
| keras23_softmax3_fetch_covtype | 실습 | 클래스 7(라벨 1~7), 대용량 데이터 |
| keras23_softmax4_digits | 실습 | 클래스 10(손글씨 숫자) |

---

## 5. 오늘 한 줄 요약

> 다중분류는 라벨을 **원-핫 인코딩**(값의 크기가 아닌 위치로 표현)한 뒤, 출력층 **softmax**(합이 1인 확률 분포, 노드 수 = 클래스 수)와 손실 **categorical_crossentropy**로 학습한다.
> 예측은 softmax 확률에서 **argmax**로 가장 큰 위치를 뽑아 판정하며, 정확도 계산 시 y_predict와 y_test 모두 argmax로 되돌려 비교한다. (이진분류의 sigmoid·binary_crossentropy·np.round에 각각 대응.)

---

## 6. 환경 메모

- fetch_covtype 등 sklearn 내장 데이터 다운로드가 SSL로 막히면 파일 맨 위에 추가:
  ```python
  import ssl
  ssl._create_default_https_context = ssl._create_unverified_context
  ```
