# 261124 복습 — 앙상블 3종: Bagging · Voting · Stacking

> **앙상블(Ensemble)은 여러 모델을 합쳐 하나보다 나은 성능을 내는** 기법이다. 오늘은 그 대표 방식 세 가지 — **Bagging**(같은 모델 여러 개 + 데이터 무작위 샘플링), **Voting**(서로 다른 모델들의 투표), **Stacking**(모델들의 예측을 다시 모델에 넣기) — 를 각각 분류(cancer)·회귀(california·diabetes) 데이터로 실습한다. 중간에 부스팅 3대장의 나머지인 **LightGBM·CatBoost**도 처음 등장한다.

## 목차
0. 앙상블이란 · 공통 세팅
1. m15 — Bagging(배깅)
2. m16 — Voting(보팅) · LightGBM·CatBoost
3. m17 — Stacking 수동 구현
4. m18 — StackingClassifier/Regressor(자동)
5. 세 기법 비교

---

## 0. 앙상블이란 · 공통 세팅

**앙상블**은 여러 모델의 예측을 합쳐 **더 정확하고 안정적인** 결과를 내는 방법이다. 대표 방식은 넷이다.

- **Bagging(배깅)**: **같은 종류 모델**을 여러 개, 각각 **데이터 일부를 무작위로 뽑아** 학습 → 평균/투표. (예: RandomForest)
- **Boosting(부스팅)**: 모델을 **순서대로 쌓아 앞 모델의 오차를 다음이 보완**. (지난주 다룸 — XGBoost 등)
- **Voting(보팅)**: **서로 다른 종류 모델**들을 묶어 **투표/평균**.
- **Stacking(스태킹)**: 모델들의 **예측 결과를 입력으로 삼아 또 다른 모델**이 최종 판단.

모든 파일의 데이터 준비는 동일한 틀이다.

```python
x, y = load_breast_cancer(return_X_y=True)   # 또는 fetch_california_housing / load_diabetes
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=444,
                                                    stratify=y)   # 분류만 stratify 사용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)       # train에 fit → 그 기준으로 test는 transform만
x_test = scaler.transform(x_test)
```

- **데이터셋 3종**: `load_breast_cancer`(유방암, **분류**)·`fetch_california_housing`(집값, **회귀**)·`load_diabetes`(당뇨 수치, **회귀**). 같은 기법을 분류·회귀 양쪽에 적용해 본다.
- **`stratify=y`는 분류에서만**: 클래스 비율을 train/test에 똑같이 유지(회귀는 연속값이라 생략).
- **스케일링**: `MinMaxScaler`를 **train에만 `fit`** 하고 test는 **`transform`만** 한다(test 정보가 새어드는 것을 막음).

---

## 1. m15 — Bagging(배깅)

**Bagging = Bootstrap Aggregating.** 같은 모델(여기선 결정 트리)을 여러 개 만들되, 각 모델에 **데이터를 무작위 복원추출(bootstrap)** 해 조금씩 다르게 학습시키고, 그 예측을 **평균(회귀)/다수결(분류)** 로 합친다. 개별 트리의 **과적합·분산을 줄여** 안정적인 결과를 얻는다.

```python
from sklearn.ensemble import BaggingRegressor

model = BaggingRegressor(
    DecisionTreeRegressor(),   # 묶을 기본 모델(약한 학습기)
    n_estimators=10,           # 모델 10개
    max_samples=0.8,           # 각 모델이 쓸 샘플 비율(80%)
    # bootstrap=True,          # 디폴트: 복원추출(중복 허용)
    bootstrap_features=False,  # 특성은 복원추출 안 함
    random_state=333, n_jobs=1,
)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))   # 회귀라 R²
```

- **`n_estimators`**: 묶을 모델 개수. **`max_samples=0.8`**: 각 모델이 전체의 80%만 뽑아 학습(서로 다른 데이터로 다양성 확보).
- **`bootstrap=True`(디폴트)**: **복원추출**(같은 샘플 중복 허용). 이 무작위성이 각 모델을 다르게 만든다. **`bootstrap_features`**: 특성(열)도 무작위로 뽑을지 여부.
- **RandomForest는 배깅의 대표**다(결정 트리 + 배깅 + 특성 무작위 선택). 배깅을 일반화하면 어떤 모델이든 묶을 수 있다.
- (보충) cancer(분류 데이터)에도 `BaggingRegressor`+`DecisionTreeRegressor`를 써서 R²로 평가하는데, 이는 **회귀기로 통일해 비교**하려는 구성이다. 분류 문제를 제대로 풀려면 `BaggingClassifier`+`DecisionTreeClassifier`로 바꾸고 정확도(acc)로 보는 것이 맞다.

---

## 2. m16 — Voting(보팅) · LightGBM·CatBoost

**Voting = 서로 다른 종류의 모델들을 묶어 투표/평균.** 배깅이 "같은 모델 여러 개"라면, 보팅은 **성격이 다른 모델들**(XGB·LightGBM·RandomForest·CatBoost)을 모아 서로의 약점을 보완한다.

```python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier      # pip install lightgbm
from catboost import CatBoostClassifier  # pip install catboost

xgb, lg, rf, cat = XGBClassifier(), LGBMClassifier(), RandomForestClassifier(), CatBoostClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat), ('LG', lg)],  # (이름, 모델) 목록
    voting='soft',     # 확률을 평균해서 결정
    # voting='hard',   # 다수결(각 모델이 낸 라벨 투표)
)
```

- **`estimators=[('이름', 모델), ...]`**: 묶을 모델들을 이름과 함께 나열한다.
- **`voting='soft'` vs `'hard'`**(분류):
  - **hard(하드 보팅)**: 각 모델이 낸 **최종 라벨을 투표**해 다수결.
  - **soft(소프트 보팅)**: 각 모델이 낸 **확률을 평균**해 가장 높은 쪽 선택. 보통 soft가 성능이 낫다(확률 정보까지 반영).
  - **회귀(`VotingRegressor`)** 는 투표 개념이 없어 **예측값을 평균**한다(voting 인자 없음).
- **새 부스팅 모델 2종**(XGBoost와 함께 "그래디언트 부스팅 3대장"):
  - **LightGBM(LGBM)**: 가볍고 빠른 부스팅. 대용량에 강함.
  - **CatBoost**: 범주형 데이터에 강하고 기본 설정으로도 성능이 좋은 부스팅.

---

## 3. m17 — Stacking 수동 구현

**Stacking = 모델들의 "예측 결과"를 새로운 입력으로 삼아, 또 다른 모델이 최종 판단.** 1차 모델 여러 개를 돌려 나온 예측들을 **새 컬럼(특성)으로 쌓아** 데이터셋을 만들고, 그걸 2차 모델(메타 모델)이 학습한다. (파일 주석: "스태킹 — 모델을 돌려 나온 걸로 컬럼을 구성해 새 데이터셋을 만듦")

```python
models = [xgb, lg, rf, cat]
train_list, test_list = [], []

for model in models:                      # ① 1차 모델들을 각각 학습
    model.fit(x_train, y_train)
    train_list.append(model.predict(x_train))   # train 예측 모으기
    test_list.append(model.predict(x_test))     # test 예측 모으기

x_train_new = np.array(train_list).T      # ② 예측들을 특성으로 (샘플수, 모델수)
x_test_new  = np.array(test_list).T       #    예: (455, 4) — 4개 모델의 예측 4열
print(x_train_new.shape)                  # (455, 4)

model2 = CatBoostClassifier(verbose=0)    # ③ 2차(메타) 모델
model2.fit(x_train_new, y_train)          #    '원본'이 아닌 '예측들'로 학습
y_pred_new = model2.predict(x_test_new)
print("스태킹 결과 : ", accuracy_score(y_test, y_pred_new))
```

- **`np.array(train_list).T`**: 모델별 예측(행)을 **전치(`.T`)** 해 **(샘플 수, 모델 수)** 형태로 만든다. 각 행이 한 샘플, 각 열이 "그 샘플에 대한 한 모델의 예측" → **메타 모델의 입력 특성**이 된다. cancer는 모델 4개라 `(455, 4)`.
- **메타 모델(`model2`)** 은 원본 데이터가 아니라 **"1차 모델들의 예측"을 보고** 최종 결정한다. "여러 전문가의 의견을 모아, 상사가 종합 판단"하는 구조다.
- **`model.__class__.__name__`**: 모델의 클래스 이름 문자열(예: `'XGBClassifier'`). 결과를 모델별로 보기 좋게 출력할 때 쓴다.
- (보충) 이 수동 버전은 **train 예측을 그대로 메타 입력으로** 쓴다. 1차 모델이 이미 본 데이터라 과하게 잘 맞아 **데이터 누수(leakage)** 위험이 있다. 이를 막는 교차검증 방식이 다음 파일의 `StackingClassifier`다.

---

## 4. m18 — StackingClassifier/Regressor(자동)

sklearn의 **`StackingClassifier`/`StackingRegressor`** 는 위 과정을 **교차검증으로 안전하게 자동화**한다.

```python
from sklearn.ensemble import StackingClassifier

model = StackingClassifier(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat), ('LG', lg)],  # 1차 모델들
    final_estimator=CatBoostClassifier(verbose=0),  # 2차(메타) 모델
    cv=5,          # ★ 교차검증으로 1차 예측 생성 (누수 방지)
    n_jobs=1,
)
model.fit(x_train, y_train)
print(accuracy_score(y_test, model.predict(x_test)))
```

- **`estimators`**(1차 모델들) + **`final_estimator`**(메타 모델)를 지정하면, 내부에서 1차 예측 생성 → 메타 학습까지 한 번에 처리한다.
- **`cv=5`**: 1차 모델의 예측을 만들 때 **5겹 교차검증**을 써, 각 샘플의 예측을 "그 샘플을 학습에 쓰지 않은 모델"로 생성한다 → m17의 데이터 누수 문제를 막는다. (StackingClassifier를 쓰는 핵심 이유)
- `StackingRegressor`도 같은 구조이며 회귀 데이터(california·diabetes)에 R²로 평가한다.

---

## 5. 세 기법 비교

| 기법 | 묶는 방식 | 모델 구성 | 합치는 방법 |
|---|---|---|---|
| **Bagging** | 데이터를 무작위 샘플링해 **병렬** 학습 | **같은** 모델 여러 개 | 평균/다수결 (RandomForest) |
| **Boosting** | 앞 모델 오차를 다음이 **순차** 보완 | 같은 모델 순차 | 가중합 (XGB·LGBM·CatBoost) |
| **Voting** | 같은 데이터로 각자 학습 | **서로 다른** 모델들 | 투표(hard)/확률평균(soft) |
| **Stacking** | 1차 예측을 특성으로 재학습 | 1차 여러 + 2차 메타 | 메타 모델이 최종 결정 |

- **공통 목적**: 한 모델의 약점(과적합·편향)을 **여러 모델로 보완**해 더 안정적인 성능을 낸다.
- **차이**: 배깅은 "같은 모델 + 데이터 다양성", 보팅은 "다른 모델들의 투표", 스태킹은 "예측을 다시 모델에 넣는 2층 구조". 보팅·스태킹에서 **LightGBM·CatBoost**가 XGBoost와 함께 주력 모델로 쓰인다.

---

## 한 줄 요약

> **앙상블**은 여러 모델을 합쳐 성능을 높이는 기법으로, 오늘 세 가지를 실습했다. **Bagging**(`BaggingRegressor`)은 **같은 모델**을 데이터 무작위 복원추출(`bootstrap`, `max_samples`)로 여러 개 학습해 평균/투표하며 RandomForest가 대표다. **Voting**(`VotingClassifier`)은 **서로 다른 모델**(XGB·LightGBM·RandomForest·CatBoost)을 묶어 **hard(다수결)/soft(확률평균)** 로 합친다(회귀는 평균). **Stacking**은 1차 모델들의 **예측을 새 특성으로 만들어 메타 모델이 최종 판단**하는 2층 구조로, 수동 구현(m17: `np.array(list).T`로 예측을 쌓음)은 데이터 누수 위험이 있어 **`StackingClassifier`의 `cv=5` 교차검증**(m18)으로 안전하게 자동화한다. 이번에 **LightGBM·CatBoost**가 XGBoost와 함께 그래디언트 부스팅 3대장으로 등장했다.
