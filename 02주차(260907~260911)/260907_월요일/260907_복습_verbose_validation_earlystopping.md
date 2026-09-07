# 2026.09.07 복습 — verbose · validation(3등분) · 학습시간 측정 · History/그래프 · EarlyStopping

> 오늘 흐름: 훈련 과정 출력 제어(verbose) → 훈련 중 검증을 위한 데이터 3등분(validation) → 실전 데이터에 validation 적용 → 학습 시간 측정 → `model.fit`의 반환값(History)으로 loss 그래프 그리기 → 과적합 전에 자동으로 멈추는 EarlyStopping.

---

## 0. 지난주 복기

- 딥러닝 4단계: 데이터 → 모델 구성 → 컴파일·훈련 → 평가·예측.
- 데이터가 많을수록 대체로 성능이 좋아진다. 순차적으로 층을 쌓는 모델이 Sequential.
- `.T`(transpose) = 행과 열을 뒤집어 shape를 맞추는 연산. `range(10)` = 0~9(0부터 10−1까지).
- `train_test_split`으로 데이터를 나누고, `scatter`로 시각화한다.
- 평가지표: mse는 제곱 때문에 값이 커지는 문제가 있고, 이를 루트로 되돌린 것이 rmse. R²는 loss만으로는 좋고 나쁨의 절대 기준이 없어 쓰는 보조지표.

---

## 1. keras15_verbose — 훈련 과정 출력 제어(verbose)

**verbose**는 사전적으로 "말이 많은"이라는 뜻이다. `model.fit`에서 **훈련 과정을 터미널에 얼마나 자세히 출력할지**를 정하는 옵션이다.

```python
model.fit(x_train, y_train, epochs=100, batch_size=2, verbose=1)
```

출력 단계(이 실습 기준):
- **verbose=0**: 침묵. 훈련 과정을 출력하지 않는다.
- **verbose=1**: 전부 출력(프로그래스바 포함). `fit`의 기본값이다.
- **verbose=2**: 프로그래스바 없이 epoch당 한 줄로 출력.
- **verbose=3 (및 그 외)**: Epoch 번호만 출력.

- 훈련 규모가 커지면 매 epoch 출력을 화면에 찍는 것도 자원 낭비가 된다. 출력을 끄면(0) 그만큼 **속도가 빨라진다.**
- 다만 출력을 끄면 진행 상황을 볼 수 없으므로, 무조건 끄는 것이 좋은 것은 아니다. 상황에 따라 선택한다.

---

## 2. keras16_validation — 데이터 3등분(train / val / test)

**validation(검증)** 은 사전적으로 "유효성 확인"을 뜻한다. 지금까지는 데이터를 train/test 2등분해서 **훈련이 끝난 뒤에만** test로 평가했다. 그러나 **훈련 도중에도** 성능을 점검하는 것이 좋다. 

과적합이 언제 시작되는지 실시간으로 보이고, 그래야 적절한 시점에 멈출 수 있기 때문이다.
- epoch을 1000으로 돌렸는데, 알고보니 200에서 이미 과적합이 시작되었다고 가정.
- train, test 2등분만 쓰면 그 사실을 끝까지 다 돌리고 나서야 알게 됨

**여기서 생긴 궁금증:** val 영역을 나눈다고 해도, 언제부터 망가진지는 결국 실행이 끝나봐야 아는거 아닌가? 1에폭이 오래 걸리는 작업은 패널에서 눈으로 따라가면서 판단할 수 있다고 쳐도, 현실적으론 하루종일 패널 보고 있지도 않을 것이고, 실행 도중 다른 작업을 하고 있을 수도 있지않나.
- val을 나누는건 "사람이 눈으로 보려고"가 아닌 "기계(코드)가 자동으로 판단하게 하려고" 이다.
- 이게 val 다음에 나올 **EarlyStopping** 과 이어짐.
- val **없음** = epoch 1000 박아놓고 자리비움 -> 돌아오니 다 끝나 있음 -> 그래프 보니 200부터 과적합이었음 -> **이미 늦음, 다시 돌려야함**
- val **있음** = epochs 5000000 박아놓고 자리 비움 → val_loss가 안 좋아지자 코드가 알아서 225쯤에서 멈춤 + 최적 지점 복원 → **돌아오니 이미 최적 상태로 완료돼 있음**

그래서 데이터를 train, val, test 세 덩어리로 나누는 것이다.

- **train**: 훈련용(가중치를 학습).
- **val(validation)**: 훈련 중간중간 성능을 점검하는 용도.
- **test**: 훈련이 모두 끝난 뒤 최종 평가용.

**val과 test의 차이:** val과 test는 둘 다 훈련(fit)에는 쓰이지 않는 평가용 데이터다. 그러나 보는 시점과 목적이 다르다.

| 구분 | val (검증) | test (평가) |
|---|---|---|
| 언제 보나 | 훈련 중 매 epoch마다 | 훈련이 완전히 끝난 뒤 한 번 |
| 목적 | 훈련 과정 감시 (과적합 확인, EarlyStopping) | 최종 성적 매기기 |
| 튜닝 영향 | 준다 (val을 보고 멈추거나 조정) | 안 준다 (건드리지 않음) |

test를 훈련 중 점검에 쓰면 그 데이터에 맞춰 튜닝하게 되어 최종 평가가 오염되므로, 감시 전용인 val을 따로 둔다.

훈련은 **epoch → validation → epoch → validation** 순으로 돌아간다. 즉 한 epoch 학습할 때마다 val 데이터로 중간 점검을 한다. 이때 나오는 손실이 **val_loss**다.

- **val_loss는 훈련(가중치 갱신)에 직접 영향을 주지 않는다.** 검증은 상태를 지켜보기만 할 뿐, 그 결과로 가중치를 바꾸지는 않는다.
- 통상적으로 val_loss가 loss보다 다소 나쁘게(높게) 나온다. val은 모델이 학습에 쓰지 않은 데이터이기 때문이다. (다만 데이터가 어떻게 나뉘느냐에 따라 더 낮게 나올 수도 있다.)

### 방법 1 — val을 직접 지정 (validation_data) · keras16_validation1
```python
x_train = np.array([1,2,3,4,5,6]); x_val = np.array([7,8]); x_test = np.array([9,10])
model.fit(x_train, y_train, epochs=10, batch_size=2,
          verbose=1,
          validation_data=(x_val, y_val))   # 검증용 데이터를 직접 넘김
```
- **`validation_data`**: 훈련 중 검증에 쓸 데이터를 `(x_val, y_val)` 형태로 직접 지정한다. `model.fit`에서 사용한다.

### 방법 2 — 슬라이싱으로 8/4/4 · keras16_validation2
```python
x = np.array(range(1, 17))
x_train = x[:8]      # 앞 8개
x_val   = x[8:12]    # 다음 4개
x_test  = x[12:]     # 마지막 4개
print(x_train.shape, x_val.shape, x_test.shape)   # (8,) (4,) (4,)
```
- 배열을 직접 잘라 train 8 / val 4 / test 4로 나눈다.

### 방법 3 — train_test_split을 두 번 · keras16_validation3
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1111)
x_val,   x_test, y_val,   y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1111)
```
- 먼저 전체를 반(8:8)으로 나눈 뒤, 그 나머지(8)를 다시 반(4:4)으로 나눠 val과 test로 만든다. 결과는 8/4/4.

### 방법 4 — 훈련에서 자동으로 val 분리 (validation_split) · keras16_validation4_split
```python
model.fit(x_train, y_train, epochs=10, batch_size=2,
          verbose=1,
          validation_split=0.33)   # x_train의 33%를 val로 자동 사용
```
- **`validation_split`**: 별도로 x_val을 만들지 않고, **훈련 데이터(x_train)에서 지정한 비율만큼을 자동으로 떼어** 검증에 쓴다.
- `validation_split=0.33`이면 훈련 데이터의 **33%(0.33 비율)** 를 val로 사용한다.
- val을 따로 준비할 필요가 없어 간편하다.

---

## 3. keras17_val1 ~ val5 — 실전 데이터에 validation 적용

앞서 배운 validation을 실전 데이터셋에 그대로 적용하는 실습이다.

| 파일 | 데이터 |
|---|---|
| keras17_val1 | california |
| keras17_val2 | diabetes |
| keras17_val3 | boston |
| keras17_val4 | dacon 따릉이 |
| keras17_val5 | kaggle 자전거 |

- 각 데이터셋의 `model.fit`에 `validation_split`(또는 `validation_data`)을 추가해 훈련 중 val_loss를 확인한다.

---

## 4. keras18_time — 학습 시간 측정(time 모듈)

AI 학습은 얼마나 걸리는지 확인하는 것이 중요하다. 훈련 전후의 시각을 재서 걸린 시간을 계산한다.

```python
import time

start_time = time.time()                      # 시작 시각
model.fit(x_train, y_train, epochs=2, batch_size=15, validation_split=0.5)
end_time = time.time()                         # 끝난 시각

print("훈련에 걸린 시간: ", round(end_time - start_time, 2), "초")
```
- **`time.time()`**: 현재 시각을 초 단위 숫자로 반환한다.
- 끝난 시각 − 시작 시각 = **훈련에 걸린 시간(초)**.
- **`round(값, 2)`**: 소수점 둘째 자리까지 반올림한다.

---

## 5. keras19_overfit1 ~ overfit5 — fit의 반환값(History)과 그래프

프로그래스바·loss·val_loss는 따로 출력하지 않아도 터미널에 찍힌다. 이는 `model.fit`이 **반환값을 가지고 있기** 때문이다. 이 반환값을 변수에 받아 활용한다.

```python
hist = model.fit(x_train, y_train, verbose=1, epochs=50, batch_size=32, validation_split=0.2)
```

### History 객체
```python
print(hist)
# <keras.src.callbacks.history.History object at 0x...>   ← 랩핑(포장)된 상태로 출력
```
- **`model.fit`의 반환값은 History(이력) 객체**다. 훈련 중 기록을 담고 있다.

### hist.history — 딕셔너리
```python
print(hist.history)
# {'loss': [414.15, 482.72, ...], 'val_loss': [45.24, 1.01, ...]}
```
- **`hist.history`**: 훈련 기록을 **딕셔너리(dictionary)** 형태로 담는다.
- **딕셔너리 = 키(key) + 값(value)** 구조. 여기서는 `'loss'`, `'val_loss'`가 키이고, epoch별 손실 값들의 **리스트**가 값이다.

### 특정 값만 꺼내기
```python
print(hist.history['loss'])       # loss 리스트만
print(hist.history['val_loss'])   # val_loss 리스트만
```
- 딕셔너리에서 키(`'loss'`)를 지정하면 그 값(리스트)만 꺼낼 수 있다.

### loss / val_loss 그래프
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트(안 넣으면 그래프 한글이 깨짐)

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:], c='red', label='loss')
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
plt.legend(loc='upper right')   # 범례를 우측 상단에 표시 # legend: 범례 / loc: 위치(location) / 'upper right': 우측상단 / 범례 표시 글자는 plt.plot 의 label에서 가져옴
plt.title('캘리포니아 loss')
plt.xlabel('epoch'); plt.ylabel('loss')
plt.grid()   # 격자 표시
plt.show()
```
- loss(빨강)와 val_loss(파랑)를 함께 그려 훈련 흐름을 눈으로 본다.
- **`[2:]`**: 리스트의 **앞 2개(초반 epoch)를 제외하고 3번째부터** 그린다. 초반 epoch의 loss가 매우 커서 그래프 y축이 눌리기 때문에, 앞부분을 잘라내 관심 구간을 크게 본다.
- 이 그래프에서 loss는 계속 내려가는데 val_loss가 어느 지점부터 올라가며 벌어지면 **과적합**이 시작된 것이다.

---

## 6. keras20_early_stopping1 — 자동 중단(EarlyStopping)

val_loss가 최소값을 찍은 뒤 더 이상 갱신되지 않는데도 epoch를 계속 도는 것은 비효율이며, 과적합으로 이어진다. 그래서 **개선이 멈추면 훈련을 자동으로 중단**한다.

**EarlyStopping(조기 종료)**: "일정 횟수(patience) 안에 최소값이 갱신되지 않으면 훈련을 멈춰라"를 수행하는 **콜백(callback)**이다. 콜백은 훈련 중 특정 시점에 자동으로 호출되는 기능을 뜻한다.

**EarlyStopping**은 #3. 컴파일, 훈련 단계에서 사용한다.

```python
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',          # 무엇을 지켜볼지
    mode = 'min',                  # 최소값을 목표로 (loss는 낮을수록 좋으므로 min)
    patience = 25,                 # 개선이 없어도 몇 epoch까지 참을지
    restore_best_weights = True,   # 중단 시 가장 좋았던 지점의 가중치로 되돌림
)

hist = model.fit(x_train, y_train,
                 verbose=1,
                 epochs=5000000,          # 아주 크게 두고 EarlyStopping이 멈추게 함
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[es])          # 콜백을 리스트로 전달
```

각 설정의 의미:
- **`monitor='val_loss'`**: 감시할 지표. 여기서는 검증 손실.
- **`mode='min'`**: 지표가 작을수록 좋은 경우 최소값을 기준으로 판단한다. (정확도처럼 클수록 좋은 지표라면 `'max'`)
- **`patience=25`**: 최소값이 갱신되지 않아도 25 epoch까지 기다린다. 그 안에 개선이 없으면 중단한다. (숫자로 넣어야 하며 따옴표를 붙이면 안 된다.)
- **`restore_best_weights=True`**: 중단 시점의 가중치가 아니라, 훈련 중 **가장 성능이 좋았던 epoch의 가중치로 되돌린다.**
- **`callbacks=[es]`**: `model.fit`에 콜백을 **리스트**로 넘긴다. 리스트이므로 이후 다른 콜백 기능도 함께 넣을 수 있다.
- **`epochs`를 아주 크게(5000000)** 두는 이유: 어차피 EarlyStopping이 적절한 시점에 멈추므로, 상한을 크게 잡아 두고 종료 판단을 EarlyStopping에 맡긴다.

---

## 7. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras15_verbose | 출력 제어 | verbose 0/1/2/3, 끄면 빨라짐 |
| keras16_validation1 | 검증 | validation_data로 val 직접 지정 |
| keras16_validation2 | 3등분 | 슬라이싱 8/4/4 |
| keras16_validation3 | 3등분 | train_test_split 두 번 |
| keras16_validation4_split | 검증 | validation_split로 훈련에서 자동 분리 |
| keras17_val1~5 | 실전 적용 | california·diabetes·boston·따릉이·자전거에 validation |
| keras18_time | 시간 측정 | time.time()로 훈련 시간 계산 |
| keras19_overfit1~5 | 반환값·그래프 | History 객체, hist.history 딕셔너리, loss 그래프 |
| keras20_early_stopping1 | 자동 중단 | EarlyStopping 콜백 (monitor·patience·restore_best_weights) |

---

## 8. 오늘 한 줄 요약

> 훈련 출력을 **verbose**로 조절하고, 훈련 중 성능을 점검하려 데이터를 **train/val/test로 3등분**한다(직접 지정 `validation_data`, 자동 분리 `validation_split`).
> `model.fit`의 반환값인 **History**(`hist.history` 딕셔너리)로 loss·val_loss를 그래프로 확인하고, val_loss가 더 개선되지 않으면 **EarlyStopping**으로 훈련을 자동 중단해 과적합을 막는다.

---

## 9. 실습 환경 메모

- `keras17_val4`, `keras18_time` 등은 따릉이·캐글 csv를 쓰므로 `_data/ddarung/`, `_data/kaggle_bike/` 경로와 pandas가 필요하다.
- 그래프에 한글을 쓰려면 `plt.rcParams['font.family'] = 'Malgun Gothic'`을 그래프 코드 위에 둔다.
