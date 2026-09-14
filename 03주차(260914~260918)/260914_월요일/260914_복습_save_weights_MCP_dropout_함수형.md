# 2026.09.14 복습 — save_weights · ModelCheckpoint · Dropout · 함수형 모델

> 오늘 흐름: 모델 저장을 가중치만 저장(save_weights)과 훈련 중 최고 지점 자동 저장(ModelCheckpoint)으로 확장 → 과적합을 줄이는 Dropout → Sequential 대신 복잡한 구조를 만들 수 있는 함수형(Functional) 모델.

---

## 1. keras29_5 / 6 — save_weights / load_weights (가중치만 저장)

지난주 `model.save`는 모델 전체(구조 + 가중치)를 저장했다. **`save_weights`는 가중치만** 저장한다.

```python
model.save_weights(path + 'keras29_5_save_weights2.weights.h5')   # 가중치만 저장 (.h5)
```
```python
# 불러올 때
model = Sequential()          # 모델 구조를 코드에 먼저 만들어야 함
model.add(Dense(40, ...)) ...
model.load_weights(path + 'keras29_5_save_weights2.weights.h5')   # 가중치만 불러옴
```
- **`save_weights`**: 가중치만 `.weights.h5` 파일로 저장한다. 모델 구조는 저장되지 않는다.
- **`load_weights`**: 저장된 가중치를 불러오지만, **모델 구조가 코드에 이미 정의되어 있어야** 실행된다. (구조 없이 가중치만으로는 실행되지 않는다.)
- 저장 시점의 의미는 model.save와 동일하다: 훈련 전에 저장하면 초기 가중치, 훈련 후에 저장하면 학습된 가중치가 담긴다.

---

## 2. keras30 — ModelCheckpoint(MCP): 최고 지점 자동 저장

**ModelCheckpoint**는 훈련 도중 **성능이 가장 좋은(val_loss가 가장 낮은) 지점의 모델을 파일로 자동 저장**하는 콜백이다.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto',
    save_best_only=True,                        # 개선될 때만(최고 지점만) 저장
    filepath=path + 'keras30_mcp1.keras',
    verbose=1,
)
model.fit(..., callbacks=[es, mcp])             # EarlyStopping과 함께 리스트로 전달
```
- **`save_best_only=True`**: val_loss가 갱신(개선)될 때만 저장해 **최고 지점 하나만** 남긴다.
- **`.keras`로 저장** → 가중치만이 아니라 **모델 전체(구조 + 가중치)** 가 저장된다.
- EarlyStopping과 차이: `restore_best_weights=True`는 훈련이 끝난 뒤 **메모리상 모델**을 최고 지점으로 되돌리는 것이고, ModelCheckpoint는 최고 지점을 **파일로 저장**하는 것이다. 둘은 역할이 달라 함께 쓴다.

### keras30_ModelCheckpoint2 — 불러와서 확인
```python
#2. 모델구성
model = load_model(path + 'keras30_mcp1.keras')   # MCP가 저장한 모델을 불러옴
# (모델 구성과 컴파일·훈련은 통째로 주석 처리)

#4. 평가, 예측 → 저장 당시와 동일한 loss·RMSE가 나온다
```
- MCP는 전체 모델을 저장하므로, 불러와서 훈련 없이 평가·예측만 해도 **저장 당시와 같은 결과**가 나온다.

### keras30_ModelCheckpoint3 — 저장 파일명에 정보 남기기
매 최고 갱신 지점을 추적할 수 있도록, 파일명에 날짜·epoch·val_loss를 넣는다.
```python
import datetime
date = datetime.datetime.now()            # 현재 시각 (예: 2026-09-14 11:40:56)
date = date.strftime("%m%d_%H%M")         # 지정 형식의 문자열로 (예: 0914_1147)

filename = '{epoch:04d}-{val_loss:.4f}.keras'   # epoch(4자리)·val_loss(소수 4자리) 자동 삽입
filepath = ''.join([path, "k30_", date, "-", filename])
# 예: ./_save/keras30/k30_0914_1147-0530-0.2812.keras
```
- **`strftime`**: 시각을 지정한 형식의 문자열로 바꾼다.
- `{epoch:04d}`, `{val_loss:.4f}`는 저장 시점의 epoch와 val_loss가 자동으로 채워지는 자리다. 파일명만 봐도 언제·몇 epoch·어떤 성능의 모델인지 구분된다.

### 모델 저장 3방식 비교
| 방식 | 저장 대상 | 확장자 | 불러올 때 |
|---|---|---|---|
| `model.save()` | 전체 모델(구조 + 가중치) | .keras | `load_model`만으로 완성 |
| `model.save_weights()` | 가중치만 | .weights.h5 | 모델 구조를 코드에 먼저 만들어야 함 |
| `ModelCheckpoint` | 훈련 중 **최고 지점**의 전체 모델(자동) | .keras | `load_model`만으로 완성 |

---

## 3. keras31 / 32 — MCP를 여러 데이터셋에 적용(save / load)

- **keras31_MCP_save_**: 각 데이터셋(diabetes, boston, 따릉이, 자전거, cancer, santander, wine, covtype, digits)을 훈련하고 ModelCheckpoint로 최고 지점을 저장한다.
- **keras32_MCP_load_**: 저장된 MCP 모델을 훈련 없이 불러와 평가·예측한다. 저장된 값이 그대로 재현되는지 확인하는 흐름이다.
- (일부 실습 파일은 미완성·오류가 섞여 있으므로, 핵심은 "훈련 후 MCP 저장 → 훈련 없이 불러와 평가"라는 패턴이다.)

---

## 4. keras33_dropout — Dropout(과적합 방지)

**Dropout**은 학습할 때 **일부 노드를 무작위로 제외하고** 연산하는 기법이다.

```python
from tensorflow.keras.layers import Dropout

model.add(Dense(40, activation='relu', input_dim=8))
model.add(Dropout(0.2))   # 바로 위 Dense(40)의 노드 중 20%를 랜덤하게 끔
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))   # 위 Dense(40)의 노드 중 30%를 끔
...
```
- **`Dropout(0.2)`**: 바로 앞 층 노드의 **20%를 무작위로 제외**한다(숫자는 제외 비율).
- **매 epoch마다 꺼지는 노드가 랜덤하게 바뀐다.**
- 효과: 모델이 특정 노드에 과하게 의존하는 것을 막아 **과적합을 줄인다.** 그 결과 성능(일반화)이 좋아지는 경우가 있으며, 매 스텝의 연산량도 줄어든다.
- **중요: Dropout은 훈련(fit) 중에만 적용된다.** 평가(evaluate)·예측(predict)할 때는 **모든 노드를 사용**한다. (학습 때만 노드를 끄고, 실제 예측 때는 온전한 모델을 쓴다.)

---

## 5. keras34_hamsu — 함수형(Functional) 모델

### 왜 함수형인가
Sequential은 층을 위에서 아래로 **일렬로만** 쌓는다. 다중 입력·출력이나 분기 같은 복잡한 구조(예: 트랜스포머)는 Sequential로 만들 수 없어, **함수형 모델**을 쓴다.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input

input1  = Input(shape=(8,))                 # 입력을 Input 레이어로 정의
dense1  = Dense(40, activation='relu')(input1)   # 층을 함수처럼 호출, 앞 층을 인자로 연결
drop1   = Dropout(0.2)(dense1)
dense2  = Dense(40, activation='relu')(drop1)
...
output1 = Dense(1)(dense6)
model   = Model(inputs=input1, outputs=output1)  # 입력과 출력을 지정해 모델로 묶음
```
- **`Input(shape=(8,))`**: 입력 형태를 정의하는 레이어(함수형에서는 입력을 층으로 명시한다).
- 각 층을 `층(...)(앞_층)` 형태로 **함수처럼 호출**해 앞 층에 연결한다.
- 마지막에 **`Model(inputs=..., outputs=...)`** 로 입력과 출력을 지정해 하나의 모델로 만든다.
- 층에 `name="ys1"`처럼 이름을 붙일 수 있고, `summary()`에서 그 이름이 표시된다(선택 사항).

### Sequential과 함수형은 표현만 다르다
- 같은 구조를 Sequential로 짜든 함수형으로 짜든 **파라미터 수와 성능은 동일**하다. 표현(작성) 방식만 다를 뿐이다. `model.summary()`로 파라미터 수가 같은지 확인한다.

---

## 6. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras29_5 / 6 | save_weights | 가중치만 저장(.h5), 불러올 땐 모델 구조 필요 |
| keras30_ModelCheckpoint1~3 | MCP | 최고 지점 자동 저장(.keras 전체 모델), 파일명에 날짜·epoch·val_loss |
| keras31 / 32 | MCP 적용 | 훈련 후 저장 → 훈련 없이 불러와 평가 |
| keras33_dropout01~10 | Dropout | 노드 일부 랜덤 제외로 과적합 방지(훈련 중에만) |
| keras34_hamsu00~10 | 함수형 모델 | Input·Model로 복잡한 구조 설계, Sequential과 성능 동일 |

---

## 7. 오늘 한 줄 요약

> 모델 저장은 전체 저장(`model.save`, .keras)·가중치만 저장(`save_weights`, .h5)·훈련 중 최고 지점 자동 저장(`ModelCheckpoint`, .keras)으로 나뉜다. save_weights는 불러올 때 모델 구조가 코드에 있어야 하고, MCP와 model.save는 `load_model`만으로 복원된다.
> **Dropout**은 노드 일부를 매 epoch 랜덤 제외해 과적합을 줄이며(훈련 중에만 적용), **함수형 모델**은 Input·Model로 Sequential이 못 만드는 복잡한 구조를 설계한다(같은 구조면 성능은 동일).
