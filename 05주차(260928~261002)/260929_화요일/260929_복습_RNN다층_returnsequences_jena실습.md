# 2026.09.29 복습 — RNN 다층 구성(return_sequences·Flatten)과 캐글 Jena 실습

> 오늘 흐름: split_x로 여러 개 예측·피처 변형(split3·4) → DNN/RNN/CNN 차원 구조 정리 → RNN 층을 여러 개 쌓는 법(return_sequences) → RNN + Flatten → 실제 캐글 시계열 데이터(Jena 기후)로 wd(풍향) 예측.

---

## 1. keras56_split3 — 여러 개를 한 번에 예측

한 값이 아니라 **101~106 여섯 개**를 예측하는 실습이다. 모델은 "5개 입력 → 다음 1개"를 예측하므로, 여섯 개를 얻으려면 **입력 window도 여섯 개** 필요하다. 그래서 **예측 데이터에도 `split_x`를 적용**한다.

```python
a = np.array(range(1, 101))
size = 6
bbb = split_x(a, size)      # (95, 6)
x = bbb[:, :-1]             # (95, 5)  앞 5개 = 입력
y = bbb[:, -1]              # (95,)    마지막 1개 = 정답
x = x.reshape(-1, 5, 1)     # (95, 5, 1) 3차원

# 예측: 96~105로 6개 window를 만들어 101~106 예측
x_predict = np.array(range(96, 106))   # [96 ... 105]
x_predict = split_x(x_predict, 5)      # (6, 5)  ← 예측 입력도 잘라야 함
x_predict = x_predict.reshape(-1, 5, 1)
y_predict = model.predict(x_predict)   # 6개 예측
```

- **핵심**: 예측할 개수만큼 window가 필요하다. `split_x(range(96,106), 5)`가 `[96~100]→101`, `[97~101]→102` … `[101~105]→106`의 6개 window를 만든다.
- 실행 결과 loss 약 0.00098, 예측값 약 100.8·101.7·102.5·103.2·103.9·104.6 (101~106 근사).

---

## 2. keras56_split4 — reshape로 피처 개수 바꾸기

같은 수열을 **한 스텝에 값 2개씩 보는(피처 2개)** 형태로 바꾸는 실습이다. 강사 주석 `(N, 10, 1) -> (N, 5, 2)`가 핵심 지시다.

```python
a = np.array(range(1, 101))
size = 11                    # 입력 10개 + 정답 1개

bbb = split_x(a, size)       # (90, 11)
x = bbb[:, :-1]              # (90, 10)  입력 10개
y = bbb[:, -1]              # (90,)     정답 1개

x = x.reshape(-1, 5, 2)      # (90, 10) → (90, 5, 2)  ★피처 2개로 재구성

model.add(LSTM(64, input_shape=(5, 2)))   # 피처 2개라 (5,2)

x_predict = np.array(range(96, 106))       # [96 ... 105] 10개 = 한 샘플분
x_predict = x_predict.reshape(1, 5, 2)     # (1, 5, 2)  ← 이미 10개라 split_x 불필요
y_predict = model.predict(x_predict)       # 106 예측
```

- **`(N,10,1) → (N,5,2)`의 의미**: 숫자는 그대로 두고 묶는 방식만 바꾼다. `[1,2,3,4,5,6,7,8,9,10]`(10스텝 1피처) → `[[1,2],[3,4],[5,6],[7,8],[9,10]]`(5스텝 2피처).
- **reshape는 split_x 뒤에 x에 적용**한다. 원본 `a`를 미리 reshape하면 split_x(1차원 전용)에 3차원이 들어가 꼬인다. **순서: ① 1차원을 split → ② 나온 x를 `(N,5,2)`로 reshape.**
- **입력이 10개(한 샘플)면 예측 데이터는 split_x 없이 바로 reshape**한다(split3처럼 여러 개 뽑는 게 아니므로).
- 실행 결과 loss 약 0.0099, 106 예측값 약 104.3.

---

## 3. DNN / RNN / CNN 차원 구조 정리

세 모델의 차원 규칙을 한 표로 정리한다.

| 구분 | DNN | RNN | CNN |
|---|---|---|---|
| **입력 데이터 차원** | 2차원 | 3차원 | 4차원 |
| **input_shape 안 숫자 개수** | 1개 | 2개 | 3개 |
| **출력 차원** | 2차원 | 2차원 | 4차원 |

- **입력 데이터 차원** (batch 포함 전체): DNN `(샘플, 피처)` 2D / RNN `(샘플, 타임스텝, 피처)` 3D / CNN `(샘플, 높이, 너비, 채널)` 4D.
- **input_shape 안 숫자 개수** (맨 앞 샘플 축 제외): DNN `input_dim=1개` / RNN `(타임스텝, 피처)=2개` / CNN `(높이, 너비, 채널)=3개`.
- **출력 차원**: DNN·RNN은 `(샘플, units)` 2D로 나와 Dense와 바로 연결. CNN은 Conv2D가 `(샘플, 높이, 너비, 필터)` 4D로 나와 Flatten이 필요.
- 이 표가 "어떤 데이터에 어떤 모델·어떤 reshape가 필요한지"를 판단하는 기준이 된다.

---

## 4. keras57_01_return_sequence — RNN 층을 여러 개 쌓기

RNN 층을 2개 이상 쌓으려 하면 그냥은 안 된다. 그 이유와 해결(`return_sequences`)을 다룬다.

### 왜 그냥 못 쌓나
- RNN 층은 기본적으로 **마지막 타임스텝의 결과 하나만** 내보낸다 → 출력이 **2차원 `(샘플, units)`**.
- 그런데 다음 RNN 층은 **3차원 입력**을 요구한다. 2차원을 주면 차원이 안 맞아 에러가 난다.

### 해결 — return_sequences=True
```python
model = Sequential()
model.add(LSTM(units=128, input_shape=(3,1), return_sequences=True))  # 3D 유지
model.add(LSTM(64, return_sequences=True))                            # 3D 유지
model.add(LSTM(32, return_sequences=True))                            # 3D 유지
model.add(LSTM(16))            # 마지막은 return_sequences 없음 → 2D로 출력
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
```
- **`return_sequences=True`**: 마지막 타임스텝만이 아니라 **모든 타임스텝의 출력을 내보낸다** → 출력이 **3차원 `(샘플, 타임스텝, units)`** 로 유지된다. 그래야 다음 RNN 층에 넣을 수 있다.
- **규칙**: RNN을 쌓을 때 **중간 층은 전부 `return_sequences=True`**, **마지막 RNN 층만 빼서(False)** 2차원으로 만들어 Dense와 연결한다.
- **성능**: 층을 여러 개 쌓는다고 무조건 좋아지지 않는다. 이 실습에서 `[50,60,70] → 약 73.1`로, 단층(keras55_2)보다 오히려 나빴다. (파일 주석: "LSTM을 여러층 쌓는다고 성능이 좋진 않음".)

---

## 5. keras57_02_rnn_flatten — RNN에 Flatten 추가

RNN 다층 출력을 Dense로 넘기는 방법으로 Flatten을 시험해본 실습이다.

```python
model.add(GRU(units=32, input_shape=(3,1), return_sequences=True))
model.add(GRU(16, return_sequences=True))
model.add(GRU(8))          # return_sequences 없음 → 2D 출력
model.add(Flatten())
model.add(Dense(32, activation='relu'))
...
model.add(Dense(1))
```
- **Flatten**: 다차원 출력을 1차원(샘플별로 일렬)으로 펴는 층. CNN에서 Conv 4D 출력을 Dense로 넘길 때 쓰던 것과 같다.
- **주의(사실 확인)**: 이 코드에서 마지막 `GRU(8)`은 `return_sequences`가 없어 이미 **2차원 `(샘플, 8)`** 로 나온다. 2차원에 Flatten을 걸면 모양이 바뀌지 않으므로 **여기서 Flatten은 사실상 아무 효과가 없다.**
- Flatten이 실제로 의미가 있으려면, 바로 앞 RNN 층을 `return_sequences=True`로 둬서 **3차원으로 나올 때** 그것을 펴는 경우다.
- 실행 결과 `[50,60,70] → 약 71.7`.

---

## 6. ★ keras58_kaggle_jena1 — 캐글 Jena 기후 데이터 실습

실제 시계열 데이터로 **풍향(wd)을 예측**하는 실습이다. 데이터가 크고 단계가 많아, 한 단계씩 나눠 정리한다.

### 과제 정의
- 데이터: `jena_climate_2009_2016.csv` — 10분 간격 기후 데이터, 약 42만 행, 14개 기후 컬럼.
- 목표: **wd (deg, 풍향)** 를 y로 하여, **2016-12-31 00:10 ~ 2017-01-01 00:00의 144개** 풍향을 예측.
- 조건: 그 144개(마지막 하루) 데이터는 **학습에서 완전히 제외**(과적합 방지).

### ① 데이터 로드
```python
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
```
- **`index_col=0`**: 첫 번째 컬럼("Date Time")을 인덱스로 지정 → 데이터 본문에서 빠진다. 그래서 남는 컬럼은 **14개 기후 값(전부 숫자)** 이고, **wd (deg)가 맨 마지막(인덱스 -1)** 컬럼이다.

### ② 시계열로 자르기 (split_x)
```python
size = 144
bbb = split_x(datasets, size)   # (420408, 144, 14)
```
- 타임스텝 144(=하루치, 10분×144=24시간)로 자른다. 결과는 `(윈도우 개수, 144, 14)` 3차원. `split_x`가 DataFrame도 그대로 잘라준다.

### ③ x, y 분리 (+ 예측 구간 제외)
```python
x = bbb[:-144, :, :-1]   # (420264, 144, 13)
y = bbb[:-144, -1, -1]   # (420264,)
```
- **축2의 `:-1`** → 마지막 컬럼(wd)을 뺀 **13개 피처**를 x로. (wd는 정답이라 입력에서 제외.)
- **`y = bbb[..., -1, -1]`** → 각 window **마지막 타임스텝의 마지막 컬럼(wd)** 을 정답으로. (인덱스는 0부터라 wd는 `-1`. 여기에 `1`을 쓰면 두 번째 컬럼 T를 예측하게 되는 실수가 나기 쉽다.)
- **축0의 `:-144`** → **마지막 144개 window를 학습에서 제외.** 이 144개가 곧 예측할 구간(2016-12-31 하루)이므로, 학습에 넣지 않아 과적합/정보 누수를 막는다.

### ④ train/test 분리
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
```

### ⑤ 스케일링 — 3차원은 2차원으로 폈다가 복구
```python
scaler = MinMaxScaler()
n_train, t, f = x_train.shape     # (N_train, 144, 13)
n_test = x_test.shape[0]

x_train = x_train.reshape(-1, f)  # 3D → 2D  (N*144, 13)
x_test  = x_test.reshape(-1, f)

scaler.fit(x_train)               # train으로만 fit
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)

x_train = x_train.reshape(n_train, t, f)   # 2D → 3D 복구
x_test  = x_test.reshape(n_test,  t, f)
```
- **왜 폈다 접나**: `MinMaxScaler`는 **2차원만** 받는다(3차원을 넣으면 "dim 3 … dim <= 2 required" 에러). 그래서 `(샘플×타임스텝, 피처13)` 2차원으로 펴서 **피처별로** 스케일한 뒤, 다시 3차원으로 되돌린다. `t`, `f`는 앞에서 `x_train.shape`를 나눠 담은 값(각각 144, 13)이다.
- **fit은 train에만**: 스케일 기준(min·max)은 훈련 데이터로만 정하고, test·예측 데이터엔 `transform`으로 그 기준만 적용한다.

### ⑥ 모델 구성·훈련
```python
model = Sequential()
model.add(LSTM(256, input_shape=(144, 13)))   # 타임스텝 144, 피처 13
model.add(Dense(256, activation='relu'))
...
model.add(Dense(1))                            # wd 값 하나 예측(회귀)

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.fit(x_train, y_train, epochs=100, batch_size=5000,
          validation_split=0.2, callbacks=[es])
```
- `input_shape`는 x의 뒤 두 축 `(144, 13)`과 일치. 회귀라 출력 `Dense(1)`, loss는 mse.
- `os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"`, `batch_size=5000`은 데이터가 커서 GPU 메모리를 아끼려는 설정이다.

### ⑦ 예측 — 학습에서 뺀 마지막 144개 window
```python
x_predict = bbb[-144:, :, :-1]    # (144, 144, 13)  예측 구간, wd 제외
y_true    = bbb[-144:, -1, -1]    # (144,)          실제 wd(정답)

x_predict = x_predict.reshape(-1, f)          # 학습 때와 같은 scaler로
x_predict = scaler.transform(x_predict)       # transform만(fit 안 함)
x_predict = x_predict.reshape(-1, t, f)       # (144, 144, 13)

y_predict = model.predict(x_predict)          # (144, 1)

r2   = r2_score(y_true, y_predict)
rmse = np.sqrt(mean_squared_error(y_true, y_predict))
```
- **`bbb[-144:]`가 곧 예측 구간**이다. 학습에서 뺀(`bbb[:-144]`) 마지막 144개 window의 마지막 타임스텝이 각각 2016-12-31 00:10 ~ 2017-01-01 00:00에 해당한다.
- 예측 데이터도 **학습에 쓴 scaler로 transform**해야 값이 맞는다(다시 fit 금지).

### ⑧ 결과와 해석 (실행값)
- 실행 결과: loss 약 6658, **R² 약 -0.13**, RMSE 약 57.2, 걸린 시간 약 2730초(≈45분).
- 예측값 144개가 대체로 **170~190 사이**에 몰려 있다(거의 일정한 값으로 찍음).
- **R²가 음수**라는 것은 "그냥 평균값으로 찍는 것보다도 못하다"는 의미다(R² 정의상 사실).
- 원인 분석(*추측*): wd는 **각도(0~360도)** 라, 예를 들어 359도와 1도가 실제로는 거의 같은 방향인데 숫자로는 358만큼 멀어서 일반 회귀(mse)로는 학습이 매우 어렵다. 그래서 모델이 안전하게 중간값 부근으로 찍는 경향이 나타난 것으로 보인다. (확정된 원인은 아니며, 각도를 sin·cos로 분해하는 등의 방법이 알려져 있으나 이 실습 범위 밖이다.)
- 이 실습의 핵심은 성능 수치보다 **대용량 실데이터를 시계열로 자르고(x/y 분리·예측 구간 제외)·스케일링(3D↔2D)·예측 구간 구성**하는 전체 파이프라인을 완성하는 데 있다.

### 실습 중 자주 났던 에러 (정리)
| 에러 | 원인 | 해결 |
|---|---|---|
| `'numpy.ndarray' object has no attribute 'drop'` | split_x가 결과를 numpy로 반환 → pandas의 `.drop` 사용 불가 | 컬럼 제거는 슬라이싱 `[:, :, :-1]`로 |
| `Found array with dim 3, while dim <= 2 required` | 3차원을 MinMaxScaler에 넣음 | `reshape(-1, f)`로 2D 변환 후 스케일, 다시 3D 복구 |
| 예측값이 엉뚱함(조용한 버그) | `y = bbb[..., 1]`로 T를 예측 | wd는 마지막 컬럼이라 `bbb[..., -1]` |
| 예측 입력 모양 안 맞음 | split 연습 코드(`[7,8,9,10]`) 잔재 | 예측 구간은 `bbb[-144:, :, :-1]`, scaler로 transform |

---

## 7. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras56_split3 | 여러 개 예측 | 예측 데이터도 split_x로 잘라 window 여러 개 생성 |
| keras56_split4 | 피처 변형 | split 후 x를 `(N,5,2)`로 reshape(=피처 2개), input_shape=(5,2) |
| (이론) 차원 구조 | DNN/RNN/CNN | 데이터 2/3/4차원, input_shape 1/2/3개, 출력 2/2/4차원 |
| keras57_01_return_sequence | RNN 다층 | 중간층 `return_sequences=True`(3D 유지), 마지막만 False |
| keras57_02_rnn_flatten | RNN+Flatten | 3D 출력을 펼 때 의미. 마지막 RNN이 2D면 Flatten은 무효과 |
| keras58_kaggle_jena1 | 캐글 실데이터 | wd 예측. 자르기→x/y분리(예측구간 제외)→3D스케일링→예측 |

---

## 8. 오늘 한 줄 요약

> RNN 층을 여러 개 쌓으려면 중간 층에 **`return_sequences=True`** 를 줘서 출력을 3차원으로 유지하고, 마지막 RNN 층만 빼서 2차원으로 만들어 Dense와 연결한다(단, 층을 쌓는다고 성능이 좋아지진 않음). 차원 규칙은 **DNN 2D·RNN 3D·CNN 4D**.
> 캐글 Jena 실습은 대용량 시계열 파이프라인 전체를 다룬다: `split_x`로 자르고 → **wd(마지막 컬럼)를 y, 나머지 13개를 x**로 분리하며 **예측 구간(마지막 144 window)은 `bbb[:-144]`로 학습에서 제외** → **3차원 데이터는 `reshape(-1, f)`로 2D로 펴서 스케일링**(train만 fit) 후 복구 → 예측은 **`bbb[-144:]`** 를 같은 scaler로 transform해 수행한다. (풍향은 각도라 일반 회귀로는 성능이 낮게 나올 수 있음 — *추측*.)
