# 261125 복습 — 모델 저장·불러오기(joblib·pickle·xgboost) · 상관계수

> 앞부분은 **학습한 모델을 파일로 저장해 두고, 재학습 없이 불러 쓰는** 세 가지 방법(joblib·pickle·xgboost 자체 저장)이다. 학습에 오래 걸리는 모델을 매번 다시 훈련하지 않으려는 실무 기능이다. 뒷부분은 데이터의 변수들이 서로 얼마나 관련 있는지 보는 **상관계수(correlation)** 와 히트맵 시각화다.

## 목차
**A. 모델 저장·불러오기**
0. 왜 저장하나 · 공통 XGB 학습(early stopping)
1. m19 — joblib
2. m20 — pickle
3. m21 — xgboost 자체 저장(save_model)
**B. 상관계수**
4. m22(1) — DataFrame을 joblib·pickle로 저장
5. m22(2) — `df.corr()` · 히트맵
6. m22(3) — 상관계수 개념 확인
7. 저장 방식 비교

---

# A. 모델 저장·불러오기

## 0. 왜 저장하나 · 공통 XGB 학습(early stopping)

**모델 저장(save)은 학습을 마친 모델을 파일로 보관하는 것**이다. 학습에 시간이 오래 걸리는 모델을, 한 번 훈련해 저장해 두면 **다음엔 재학습 없이 불러와(load) 바로 예측**할 수 있다(서비스 배포·재사용의 기본). m19·m20·m21은 **같은 XGBoost 회귀 모델을 학습한 뒤, 저장하는 방법만 다르게** 한 것이다.

```python
parameters = {'n_estimators':2000, 'learning_rate':0.05, 'max_depth':7,
              'gamma':1, 'min_child_weight':1, 'subsample':1,
              'colsample_bytree':1, 'reg_alpha':0, 'reg_lambda':1, 'random_state':3377}

model = XGBRegressor()
model.set_params(**parameters, early_stopping_rounds=20)   # 파라미터 한 번에 적용

model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],   # 조기 종료를 판단할 검증 데이터
          verbose=1)
```

- **`set_params(**parameters)`**: 딕셔너리로 모아둔 하이퍼파라미터를 **`**`(언패킹)** 로 한 번에 적용한다. 모델 생성과 파라미터 설정을 분리하는 방식이다.
- **`early_stopping_rounds=20`**: **검증 점수가 20번(라운드) 동안 나아지지 않으면 학습을 조기 종료**한다. `n_estimators=2000`으로 크게 잡아도, 성능이 더 안 오르면 알아서 멈춰 **과적합과 낭비를 막는다.** 이 판단을 위해 `fit`에 **`eval_set=[(x_test, y_test)]`**(검증 데이터)를 넘긴다.
- XGB 주요 하이퍼파라미터: `learning_rate`(학습률), `max_depth`(트리 깊이), `gamma`·`min_child_weight`(분기 조건), `subsample`·`colsample_*`(샘플·특성 비율), `reg_alpha`·`reg_lambda`(규제).

## 1. m19 — joblib

```python
import joblib
path = './_save/joblib/'
joblib.dump(model, path + 'm19_joblib1_save.joblib')     # 저장
# ---- 다른 파일에서 ----
model = joblib.load(path + 'm19_joblib1_save.joblib')    # 불러오기
print(model.score(x_test, y_test))   # 재학습 없이 바로 예측 → 저장 때와 같은 점수
```

- **`joblib.dump(객체, 경로)`** 로 저장, **`joblib.load(경로)`** 로 불러온다. 간결하다.
- joblib은 **큰 numpy 배열을 담은 객체(머신러닝 모델)를 효율적으로** 저장하도록 만들어져, sklearn 계열 모델 저장에 자주 쓴다.
- 불러온 모델은 학습을 다시 하지 않아도 저장 당시 성능(R² 약 0.819)을 그대로 낸다 — 저장·복원이 제대로 됐다는 확인.

## 2. m20 — pickle

```python
import pickle
path = './_save/pickle/'
pickle.dump(model, open(path + 'm20_pickle1_save.pickle', 'wb'))   # 저장 (wb: 바이너리 쓰기)
# ---- 다른 파일에서 ----
model = pickle.load(open(path + 'm20_pickle1_save.pickle', 'rb'))  # 불러오기 (rb: 바이너리 읽기)
```

- **pickle은 파이썬 범용 객체 직렬화** 모듈이다(모델뿐 아니라 거의 모든 파이썬 객체). 모델 저장에도 쓸 수 있다.
- joblib과 달리 **파일을 직접 열어 넘긴다**: 저장은 **`open(경로, 'wb')`**(바이너리 쓰기), 로드는 **`open(경로, 'rb')`**(바이너리 읽기). 모델은 텍스트가 아니라 바이너리라 `wb`/`rb`를 쓴다.
- 결과는 joblib과 동일(R² 약 0.819). joblib이 pickle을 기반으로 대용량 배열에 최적화한 것이라, 둘은 사실상 같은 일을 한다.

## 3. m21 — xgboost 자체 저장(save_model)

```python
path = './_save/xgb/'
model.save_model(path + 'm21_xgb1_save.json')   # XGBoost 전용 저장 (json 등)
# ---- 다른 파일에서 ----
model = XGBRegressor()              # 먼저 빈 모델을 만들고
model.load_model(path + 'm21_xgb1_save.json')   # 그 안에 불러오기
```

- **`model.save_model(경로)` / `model.load_model(경로)`**: XGBoost가 **자체 제공**하는 저장 기능. `.json`(또는 바이너리)로 저장한다.
- joblib/pickle과의 차이: **불러올 때 `XGBRegressor()`로 빈 모델을 먼저 만들고** 거기에 `load_model`로 채운다(joblib/pickle은 객체를 통째로 복원). XGBoost **자체 포맷이라 라이브러리 버전이 달라도 호환이 잘 되고**, 파이썬이 아닌 다른 환경에서도 읽기 쉽다.
- (보충) 이 파일의 로드 코드는 저장(`'./_save/xgb/....json'`)과 **경로·확장자가 어긋나** 있다(`'./_save/pickle/....dat'`를 불러오려 함). 그대로 실행하면 파일을 못 찾아 오류가 난다 — 저장과 로드의 **경로·파일명·확장자는 정확히 일치**시켜야 한다는 점만 확인하면 된다(개념 자체는 `save_model`/`load_model`).

---

# B. 상관계수

## 4. m22(1) — DataFrame을 joblib·pickle로 저장

저장 기능은 모델뿐 아니라 **데이터(DataFrame)에도** 쓸 수 있다. iris 데이터를 DataFrame으로 만들어 저장한다.

```python
from sklearn.datasets import load_iris
datasets = load_iris()
df = pd.DataFrame(datasets['data'], columns=datasets.feature_names)
df['Target'] = datasets.target        # 정답(품종)을 컬럼으로 추가

joblib.dump(df, './_save/joblib/m22_df_iris.joblib')          # 데이터도 저장 가능
pickle.dump(df, open('./_save/pickle/m22_df_iris.pickle','wb'))
```

- sklearn 데이터를 **`pd.DataFrame(데이터, columns=특성이름)`** 으로 표 형태로 만들고, 정답 열(`Target`)을 붙인다. 전처리한 데이터를 저장해 두면 다음에 바로 불러 쓸 수 있다.

## 5. m22(2) — `df.corr()` · 히트맵

**상관계수(correlation)는 두 변수가 함께 변하는 정도**를 −1~1로 나타낸 값이다. `df.corr()`로 모든 열 쌍의 상관계수 행렬을 구하고, 히트맵으로 한눈에 본다.

```python
df = joblib.load('./_save/joblib/m22_df_iris.joblib')
print(df.corr())          # 열 쌍마다 상관계수 (상관관계 행렬)

import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()
```

- **`df.corr()`**: 모든 수치형 열 쌍의 상관계수를 계산한 **정사각 행렬**. 대각선은 자기 자신이라 항상 1이다.
- **`sns.heatmap(...)`**: 상관계수 행렬을 **색의 농담으로** 보여준다. **`annot=True`**(칸마다 숫자 표시), **`square=True`**(칸을 정사각형), **`cbar=True`**(색 범례 막대). 어떤 특성이 정답(Target)과 관련 깊은지, 특성끼리 겹치는지(다중공선성) 파악에 쓴다.

## 6. m22(3) — 상관계수 개념 확인

작은 예제로 상관계수의 의미를 명확히 본다.

```python
df = pd.DataFrame({'X1':[1,2,3,4,5],          # Y와 같이 증가
                   'X2':[5,4,3,2,1],          # Y와 반대로 감소
                   'X3':[-1000,50,7,100,-1100], # 규칙 없음
                   'Y' :[2,3,4,5,6]})
sns.heatmap(df.corr(), annot=True, square=True, cbar=True); plt.show()
```

- **상관계수 읽는 법**:
  - **+1에 가까움(양의 상관)**: 한 변수가 커지면 다른 변수도 커진다. 여기서 **X1과 Y는 +1**(완벽히 같이 증가).
  - **−1에 가까움(음의 상관)**: 한 변수가 커지면 다른 변수는 작아진다. **X2와 Y는 −1**(완벽히 반대).
  - **0에 가까움(무상관)**: 뚜렷한 관계 없음. **X3는 Y와 거의 0**.
- 상관계수는 **직선적(선형) 관계의 세기와 방향**만 나타낸다. 값이 0이라도 곡선 관계는 있을 수 있고, 상관이 높다고 **인과(원인-결과)인 것은 아니다.**

---

## 7. 저장 방식 비교

| 방식 | 저장 | 불러오기 | 특징 |
|---|---|---|---|
| **joblib** | `joblib.dump(model, 경로)` | `joblib.load(경로)` | 대용량 배열(ML 모델)에 효율적, 간결 |
| **pickle** | `pickle.dump(model, open(경로,'wb'))` | `pickle.load(open(경로,'rb'))` | 파이썬 범용 객체 직렬화 |
| **xgboost** | `model.save_model('*.json')` | 빈 모델 만든 뒤 `model.load_model(...)` | XGB 자체 포맷, 버전·환경 호환성 좋음 |

- 셋 다 **"학습한 모델을 파일로 보관 → 재학습 없이 복원"** 이라는 목적은 같다. sklearn 계열은 joblib/pickle, XGBoost는 자체 `save_model`이 무난하다. **저장과 로드의 경로·확장자는 반드시 일치**시켜야 한다.

---

## 한 줄 요약

> **모델 저장·불러오기**는 학습을 마친 모델을 파일로 보관해 재학습 없이 재사용하는 기능으로, **joblib**(`dump`/`load`, 대용량 배열에 효율적)·**pickle**(`dump`/`load`에 `open(경로,'wb'/'rb')`, 파이썬 범용)·**xgboost 자체**(`save_model`/`load_model`, 불러올 땐 빈 모델 먼저 생성) 세 방식이 있고 저장·로드 경로는 정확히 일치해야 한다. 공통으로 쓴 XGB 학습에서는 `set_params(**파라미터)`로 하이퍼파라미터를 한 번에 적용하고 **`early_stopping_rounds`+`eval_set`** 으로 검증 점수가 개선되지 않으면 조기 종료했다. **상관계수**는 두 변수가 함께 변하는 정도를 −1~1로 나타낸 값(+1 양의 상관, −1 음의 상관, 0 무상관)으로, **`df.corr()`** 로 상관관계 행렬을 구해 **`seaborn.heatmap`**(`annot`·`square`·`cbar`)으로 시각화하며, 선형 관계의 세기·방향만 보여줄 뿐 인과를 뜻하지는 않는다.
