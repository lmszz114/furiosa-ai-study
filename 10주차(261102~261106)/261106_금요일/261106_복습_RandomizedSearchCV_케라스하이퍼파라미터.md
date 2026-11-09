# 261106 복습 — RandomizedSearchCV · 케라스 하이퍼파라미터 튜닝

> 지난 시간 `GridSearchCV`(모든 조합 전부 시도)에 이어, 오늘은 **무작위로 일부 조합만 골라 빠르게 탐색하는 `RandomizedSearchCV`** 를 배우고, 그 방식을 **케라스(딥러닝) 모델의 하이퍼파라미터 튜닝**에 적용한다.

## 목차
1. RandomizedSearchCV란 — GridSearchCV와 비교
2. m09_00 — 기본 (random_state · refit · n_iter)
3. m09_01_digits — Grid vs Random 속도 비교
4. m09_02_diabetes — 회귀에 적용
5. keras77 — 케라스 모델 하이퍼파라미터 튜닝
6. 한 줄 요약

---

## 1. RandomizedSearchCV란 — GridSearchCV와 비교

`GridSearchCV`는 지정한 **모든 파라미터 조합을 하나도 빠짐없이** 교차검증한다. 정확하지만, 조합이 많아지면(수백~수천) 너무 느리다. `RandomizedSearchCV`는 그 조합들 중 **무작위로 몇 개(`n_iter`개)만 뽑아** 시도한다.

| | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| 탐색 방식 | 모든 조합 완전 탐색 | 무작위로 `n_iter`개만 |
| 속도 | 느림 (조합 수 × fold) | 빠름 (`n_iter` × fold) |
| 장점 | 빠뜨림 없이 전부 확인 | 조합이 많아도 빠르게 훑음 |
| 단점 | 조합 많으면 매우 느림 | 운에 따라 최고 조합을 놓칠 수 있음 |

- **핵심**: 조합이 적으면 Grid, **조합이 많으면 Random**이 현실적이다. 무작위라 "최고"를 보장하진 않지만, 실제로는 비슷한 성능을 훨씬 빨리 찾는 경우가 많다(아래 3번 결과).

---

## 2. m09_00 — 기본 (random_state · refit · n_iter)

`GridSearchCV`를 쓰던 코드에서 모델 부분만 `RandomizedSearchCV`로 바꾼다.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)

parameters = [
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1,0.01,0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1,0.01,0.001]},                           # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1,0.01,0.001]},                     # 12
]  # 전체 42가지

model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold,
                    verbose=1,
                    n_jobs=-1,
                    random_state=333,   # 무작위 뽑기를 고정(재현용)
                    refit=True,         # 최적 조합으로 전체 train 재학습
                    n_iter=3,           # 42개 중 3개만 무작위로 시도
                    )
model.fit(x_train, y_train)

print('model.score :', model.score(x_test, y_test))
print('best_score :', model.best_score_)
```

- **`n_iter=3`**: 전체 42개 조합 중 **무작위로 3개만** 골라 교차검증한다. 이 숫자가 "몇 번 시도할지"를 정한다(클수록 더 많이 탐색 = 느리지만 꼼꼼).
- **`random_state=333`**: 무작위로 뽑는 과정을 고정해, 매번 같은 조합이 뽑히게 한다(결과 재현).
- **`refit=True`**: 최적 조합을 찾은 뒤, **그 조합으로 전체 train 데이터를 다시 학습**해 둔다. 그래서 `model.score`·`model.predict`를 바로 쓸 수 있다(최적 모델이 `best_estimator_`에 담김).

---

## 3. m09_01_digits — Grid vs Random 속도 비교

같은 데이터(digits)에 GridSearch와 RandomSearch를 각각 돌려 **속도와 성능을 비교**한다.

```python
model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold,
                    verbose=1, n_jobs=-1)   # n_iter 기본값 10

model.fit(x_train, y_train)
print("최적의 파라미터 :", model.best_params_)
print('model.score :', model.score(x_test, y_test))
print('best_score :', model.best_score_)
```

실측 결과(주석):
```
# ===== GridSearch =====      # ===== RandomSearch =====
# model.score : 0.9333        # model.score : 0.9333
# best_score  : 0.9645        # best_score  : 0.9645
# 걸린시간    : 14.64 초       # 걸린시간    : 6.82 초
```

- **성능은 같은데(0.9333 / 0.9645) 시간은 절반 이하(14.64초 → 6.82초).** 모든 조합을 다 보지 않고 일부만 무작위로 봐도 **이 경우엔 최고 조합을 놓치지 않으면서 훨씬 빨랐다.** RandomSearch의 효율을 보여주는 예다.
- (이 파일은 "파라미터 튜닝하면서 돌려보기"라, `parameters` 값들을 이것저것 바꿔 실험한 흔적이 있다. 값 자체보다 **Grid↔Random 비교**가 요점이다.)

---

## 4. m09_02_diabetes — 회귀에 적용

분류(digits)에서 **회귀(diabetes, 당뇨 진행도)** 로 바꿔 똑같이 RandomSearch를 돌린다. 회귀라서 바뀌는 지점들이 있다.

```python
from xgboost import XGBRegressor      # 회귀용 모델
from sklearn.model_selection import KFold   # StratifiedKFold(X) → KFold

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333,
    # stratify=y,   # ★ 회귀에서는 사용 불가 (연속값이라 클래스 개념이 없음)
)

kfold = KFold(n_splits=5, shuffle=True, random_state=333)   # 회귀 → 일반 KFold

model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1, n_jobs=-1)
model.fit(x_train, y_train)

print('model.score :', model.score(x_test, y_test))   # 회귀는 R² 점수
y_predict = model.predict(x_test)
print('r2_score :', r2_score(y_test, y_predict))
```

- **회귀에서 바뀌는 3가지**(지난 번에도 나온 원칙):
  - 모델: `XGBClassifier` → **`XGBRegressor`**
  - 교차검증: `StratifiedKFold` → **`KFold`** (연속값은 클래스 비율을 맞출 수 없음)
  - `stratify=y` 제거, 평가지표는 **R²** (`model.score`도 R²로 나옴)
- GridSearch와 RandomSearch 결과가 같고(R² 0.352), 시간은 RandomSearch가 더 짧다(8.96초 → 5.71초).

---

## 5. keras77 — 케라스 모델 하이퍼파라미터 튜닝

지금까지는 사이킷런 모델(XGB 등)을 튜닝했다. 이번엔 **케라스(딥러닝) 모델**을 RandomizedSearchCV로 튜닝한다. 딥러닝은 조절할 파라미터(층 수·노드 수·드롭아웃·옵티마이저·배치 등)가 **매우 많아** 완전 탐색(Grid)이 비현실적이라, RandomSearch가 특히 잘 맞는다.

### 5-1. 모델을 "함수"로 만들기

튜닝하려면 파라미터를 바꿔가며 모델을 새로 만들 수 있어야 한다. 그래서 **모델 생성을 함수로** 감싼다.

```python
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    # ... hidden3·4·5 ...
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model
```

- 노드 수·드롭아웃·활성화 함수·옵티마이저 등을 **인자로 받아** 모델을 만든다. 인자를 바꾸면 다른 구조의 모델이 나온다 → 이게 튜닝 대상이 된다.

### 5-2. 탐색할 파라미터 후보 정의

```python
def create_hyperparameter():
    return {
        'batch_size' : [32, 16, 8, 1, 64],
        'optimizer' : ['adam', 'rmsprop', 'adadelta'],
        'drop' : [0.2, 0.3, 0.4, 0.5],
        'activation' : ['relu', 'elu', 'selu', 'linear'],
        'node1' : [128, 64, 32, 16],
        'node2' : [128, 64, 32, 16],
        # ... node3·4·5 ...
    }
hyperparameters = create_hyperparameter()
```

- 각 파라미터의 후보 값들을 딕셔너리로 모은다. 조합 수가 **엄청나게 많다**(배치5 × 옵티마이저3 × 드롭4 × 활성화4 × 노드4⁵ … → 수만 가지). Grid로는 못 돌리고 **Random으로 일부만** 뽑아야 한다.

### 5-3. 케라스 모델을 sklearn이 쓰도록 감싸기(wrapper)

```python
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)   # 케라스 → sklearn 호환

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3,
                           random_state=333,
                           n_iter=10,      # 무작위 10개 조합만
                           refit=True,
                           verbose=1)
```

- **`KerasRegressor(build_fn=...)`**: `RandomizedSearchCV`는 사이킷런 도구라 케라스 모델을 그대로 못 받는다. **케라스 모델을 sklearn 모델처럼 보이게 감싸는 어댑터(wrapper)** 가 `KerasRegressor`다. `build_fn`에 모델 생성 함수를 넘긴다. (분류면 `KerasClassifier`)
- 이렇게 감싸면 사이킷런의 Search에 넣어 똑같이 튜닝할 수 있다. `n_iter=10`으로 수만 조합 중 10개만 시도한다.

### 5-4. 학습 — 케라스 학습 옵션을 fit에 전달

```python
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=8)

model.fit(x_train, y_train, epochs=1000, callbacks=[es, rlr],
          validation_split=0.1, verbose=1)

print("최적의 파라미터 :", model.best_params_)
print('model.score :', model.score(x_test, y_test))
print('best_score :', model.best_score_)
```

- `fit`에 **케라스 학습 인자**(`epochs`, `callbacks`, `validation_split`)를 그대로 넘긴다. 각 조합을 학습할 때 EarlyStopping·ReduceLROnPlateau가 적용된다.
- 결과로 **최적의 하이퍼파라미터 조합**(`best_params_`)과 그 성능을 얻는다.
- (참고: `tensorflow.keras.wrappers.scikit_learn`의 `KerasRegressor`는 과거 방식이다. 최신 환경에서는 별도 패키지 `scikeras`의 `KerasRegressor`를 쓰기도 한다. 교육 환경의 버전에 맞춰 둔 코드다.)

---

## 6. 한 줄 요약

> **RandomizedSearchCV**는 GridSearchCV가 "모든 조합"을 보는 것과 달리, **무작위로 `n_iter`개 조합만** 교차검증해 훨씬 빠르게 최적값을 찾는다(digits에서 성능은 같고 시간은 절반). 조합이 많을수록 유리하며, `random_state`로 재현을 고정하고 `refit=True`로 최적 조합을 전체 train에 재학습한다. 회귀에 쓸 땐 `XGBRegressor`·`KFold`·R²로 바꾸고 `stratify`를 뺀다. **케라스(딥러닝) 모델**은 파라미터가 매우 많아 Random이 특히 맞는데, 모델을 `build_model()` 함수로 만들고 후보를 딕셔너리로 정의한 뒤 **`KerasRegressor`로 감싸(wrapper)** 사이킷런의 Search에 넣어 튜닝한다(`fit`에 epochs·callbacks 전달).
