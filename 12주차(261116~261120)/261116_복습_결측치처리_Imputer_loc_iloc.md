# 261116 복습 — 결측치 처리(보간·Imputer) · pandas loc/iloc

> 데이터 전처리의 핵심인 **결측치(NaN) 처리**를 다룬다. 단순히 지우거나 채우는 것부터(interpolate·fillna), sklearn의 Imputer 계열(Simple→KNN→Iterative)까지 점점 정교해진다. 뒷부분은 데이터를 뽑아내는 pandas의 **loc/iloc** 문법이다.

## 목차
**A. 결측치 처리**
0. 결측치 처리 방법 개요
1. pd01 — interpolate (보간)
2. pd02 — pandas 기본 결측치 처리(fillna·dropna·ffill·bfill)
3. pd03 — SimpleImputer
4. pd04 — KNNImputer (개념)
5. pd05 — IterativeImputer
**B. pandas 인덱싱**
6. pd06_1 — loc / iloc 기본
7. pd06_2 — loc / iloc 범위(슬라이싱)

---

# A. 결측치 처리

## 0. 결측치 처리 방법 개요

결측치(비어 있는 값, NaN)를 다루는 방법은 크게 이렇다. (pd01 주석 정리)

1. **삭제**: 결측치가 있는 **행 또는 열**을 통째로 버린다.
2. **임의의 값으로 채우기(fillna)**: 0 / **평균(mean, 이상치에 약함)** / **중위값(median)** / 앞값(ffill) / 뒷값(bfill) / 특정 상수 / 최빈값 등.
3. **보간(interpolate)**: 알려진 값들 사이에 새 값을 추정해 끼워 넣는다(점과 점 사이를 이어서 채움).
4. **모델 예측**: 결측치를 **별도 모델로 예측(`.predict`)** 해서 채운다. (가장 정교, Iterative Imputer가 이 방식)
5. **부스팅(트리) 계열 모델**: XGBoost 등 트리 모델은 **결측치·이상치에 영향을 덜 받아** 비교적 자유롭다(모델 자체가 결측을 처리하기도 함).

> 아래 1~5번이 이 순서대로 "단순 → 정교"하게 올라간다. 어떤 방법이 정답인지는 **데이터의 성격**을 보고 정해야 한다(예: 평균은 이상치가 있으면 왜곡됨).

---

## 1. pd01 — interpolate (보간)

**보간(interpolate)은 알려진 데이터 점들 사이에 새 점을 추정해 채우는 기법**이다. 기본은 **선형(linear)** — 앞뒤 값을 직선으로 이어 그 중간값으로 메운다(흔히 "점과 점 사이 반띵").

```python
dates = pd.to_datetime(dates, dayfirst=True)   # 문자열 → 날짜형. dayfirst=일/월/년 순으로 읽기
ts = pd.Series([0, 2, np.nan, np.nan, 8, 10, np.nan, 14, 15], index=dates)

ts = ts.interpolate()   # 결측치를 보간으로 채움
```

- **`pd.to_datetime(..., dayfirst=True)`**: `'11/11/2026'` 같은 문자열을 날짜형으로 바꾼다. `dayfirst=True`면 **일/월/년** 순으로 해석한다.
- **보간 결과의 특징**(중요):
  - **중간의 결측치**: 앞뒤 값을 이어 **선형(linear)** 으로 채워진다. 예: 2와 8 사이의 두 NaN → 4, 6.
  - **맨 끝(마지막)이 결측치**: 뒤에 이을 값이 없으면 **앞값으로(ffill처럼)** 채워진다.
  - **맨 앞(첫 번째)이 결측치**: 앞에 기준 값이 없어 **처리되지 않고 NaN으로 남는다.**
- 시계열(날짜 인덱스) 데이터에서 자연스러운 채움 방식이다.

---

## 2. pd02 — pandas 기본 결측치 처리

pandas 자체 기능으로 결측치를 확인·삭제·채우는 법을 종합한다. 실습 데이터는 일부러 군데군데 NaN을 둔 4열(x1~x4)짜리 DataFrame이다.

### 결측치 확인

```python
print(data.isnull())        # 각 칸이 결측이면 True
print(data.isnull().sum())  # 열별 결측치 '개수' ★ 자주 씀
print(data.info())          # 열별 non-null 개수·자료형 요약
```
- **`isnull().sum()`**: 열마다 결측치가 몇 개인지 한눈에 본다. 전처리 첫 단계에서 거의 항상 쓴다.

### 결측치 삭제 — dropna

```python
data.dropna()         # 디폴트 = 행(axis=0) 삭제
data.dropna(axis=0)   # 결측 있는 '행' 삭제
data.dropna(axis=1)   # 결측 있는 '열' 삭제
```
- **`axis=0`은 행, `axis=1`은 열**이다. 기본은 행 삭제. (데이터가 아까우면 삭제보다 채우기를 택함)

### 결측치 채우기 — fillna / ffill / bfill

```python
data.fillna(data.mean())     # 평균으로 (이상치 주의)
data.fillna(data.median())   # 중위값으로
data.fillna(0)               # 0으로
data.ffill()                 # 앞(위) 값으로 채움 → 첫 행은 채울 값 없어 NaN
data.bfill()                 # 뒤(아래) 값으로 채움 → 마지막 행은 NaN
```
- **`fillna(값)`**: 지정한 값/통계치로 NaN을 메운다. `mean()`·`median()`은 **열별** 통계값으로 채운다.
- **`ffill`(forward fill, 앞값)·`bfill`(backward fill, 뒷값)**: 바로 위/아래 값을 끌어다 채운다. **시계열에서 많이 쓴다.** ffill은 첫 행, bfill은 마지막 행이 채울 기준이 없어 NaN으로 남을 수 있다.

### 특정 컬럼만 골라 처리

```python
means = data['x1'].mean()              # 6.5 (x1 열의 평균)
data['x1'] = data['x1'].fillna(means)  # x1만 평균으로
data['x2'] = data['x1'].ffill()        # ← (주의) x1을 ffill해서 x2에 넣음
data['x3'] = data['x3'].fillna(med)    # x3만 중위값으로
```
- 열마다 **다른 전략**으로 채울 수 있다(x1은 평균, x3는 중위값 등). 실무에선 열 성격에 따라 방법을 달리한다.
- (보충) 가운데 줄 `data['x2'] = data['x1'].ffill()`은 **x2가 아니라 x1을 ffill한 값**을 x2에 넣는다. `data['x2'].ffill()`의 오타로 보이지만 강사 코드라 그대로 둔다 — 요점은 "열별로 골라 처리한다"는 패턴이다.

---

## 3. pd03 — SimpleImputer

여기서부터는 **sklearn의 Imputer** 계열이다. pandas `fillna`와 하는 일은 비슷하지만, **sklearn 변환기(transformer) 규격**(`fit_transform`)을 따라 머신러닝 파이프라인에 그대로 끼울 수 있다는 게 장점이다.

**SimpleImputer는 "단순 통계값"으로 채우는 가장 기본 Imputer**다. `strategy`로 어떤 값으로 채울지 고른다.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()                 # 디폴트 strategy='mean' (평균)
imputer = SimpleImputer(strategy='mean')         # 평균
imputer = SimpleImputer(strategy='median')       # 중위값
imputer = SimpleImputer(strategy='most_frequent')# 최빈값(가장 많이 나온 값)
imputer = SimpleImputer(strategy='constant', fill_value=777)  # 지정 상수로

data2 = imputer.fit_transform(data)       # 채워진 결과(numpy 배열)
```

- **`fit_transform(data)`**: 데이터를 보고 채울 값을 계산(`fit`)한 뒤 실제로 채워서(`transform`) 돌려준다. 결과는 **numpy 배열**이다(DataFrame 아님).
- **strategy 종류**: `mean`(평균, 기본)·`median`(중위값)·`most_frequent`(최빈값)·`constant`(`fill_value`로 지정한 상수, 예 777).
- (보충) pd03_2에 나온 `SimpleImputer(strategy=lambda x: np.max(x))`는 **동작하지 않는다.** SimpleImputer의 strategy는 위 4개 문자열만 받고 **함수(callable)는 지원하지 않는다.** 실제로 그 코드는 그 imputer를 쓰지 않고 넘어간다(앞의 most_frequent 결과를 다시 출력). "최댓값으로 채우기는 이렇게 안 된다"는 걸 보여주는 셈이다.

---

## 4. pd04 — KNNImputer (개념)

> **참고**: 파일 이름은 `KNNImputer`지만, 안의 코드는 pd03_2와 같은 **SimpleImputer(most_frequent·constant)** 실습이다. KNNImputer를 실제로 `fit_transform` 하는 코드는 이 파일에 없다. 아래는 **개념 설명**이다.

**KNNImputer는 "가장 비슷한 이웃 샘플들"의 값으로 결측치를 채운다.** 결측이 있는 행과 **거리가 가까운 K개 행**을 찾아, 그 이웃들의 해당 열 값(평균)으로 메운다.

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)    # 가까운 이웃 5개 기준
data2 = imputer.fit_transform(data)
```

- **SimpleImputer와의 차이**: SimpleImputer는 그 **열 전체의 통계값**(평균 등)으로 일괄 채우지만, KNNImputer는 **비슷한 행끼리** 보고 채운다 → 데이터의 **관계·패턴을 반영**해 더 정교하다.
- **`n_neighbors`**: 몇 개의 이웃을 볼지(K). KNN 분류기와 같은 거리 기반 아이디어다.
- 단점: 거리 계산을 하므로 데이터가 크면 느리고, **스케일(단위)** 에 민감하다(사전 스케일링 권장).

---

## 5. pd05 — IterativeImputer

**IterativeImputer는 결측치를 "모델로 예측"해 채우는 가장 정교한 방식**이다. **다른 열들을 입력(feature)으로 써서 결측이 있는 열을 회귀 모델로 예측**하고, 이 과정을 여러 번 반복(iterative)해 값을 점점 다듬는다. (앞 개요의 "4. 모델 예측" 방식)

```python
from sklearn.experimental import enable_iterative_imputer  # 실험적 기능이라 이걸 먼저 import 해야 켜짐
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

xgb = XGBRegressor(max_depth=5, learning_rate=0.1, random_state=333)

imputer = IterativeImputer(
    estimator=xgb,     # 결측 열을 예측할 모델 (여기선 XGBoost 회귀)
    max_iter=10,       # 반복 횟수 (최대)
    # tol=0.001,       # 얼리스탑 기준 (변화가 이보다 작으면 멈춤)
    random_state=333,
)
data2 = imputer.fit_transform(data)
```

- **동작 원리**: "x4가 비었다 → x1·x2·x3로 x4를 예측하는 모델을 학습해 채운다"를 **모든 결측 열에 대해 돌리고, 전체를 여러 번 반복**해 수렴시킨다(MICE라고 부르는 방식).
- **`estimator=`**: 예측에 쓸 모델을 넣는다. 여기선 **XGBoost 회귀**를 썼다(트리 계열이라 결측·이상치에 강함 — 개요 5번).
- **`max_iter`**: 최대 반복 횟수. **`tol`**: 값 변화가 이보다 작아지면 일찍 멈추는 기준(얼리스탑과 비슷).
- 가장 정확할 수 있지만 **계산이 무겁다.** sklearn에서 아직 실험적 기능이라 `enable_iterative_imputer`를 **먼저 import** 해야 쓸 수 있다.
- (보충) 코드의 `leaning_rate`는 `learning_rate`의 오타다. XGBoost는 모르는 인자를 무시하거나 경고만 내서 실행은 되지만, 의도대로 학습률이 반영되진 않는다(강사 코드라 그대로 두되 알아둘 것).

### Imputer 3종 정리

| Imputer | 채우는 방식 | 특징 |
|---|---|---|
| **SimpleImputer** | 열의 통계값(평균·중위·최빈·상수) | 가장 단순·빠름, 관계 무시 |
| **KNNImputer** | 가까운 이웃 K개 샘플의 값 | 행 간 유사성 반영, 스케일 민감 |
| **IterativeImputer** | 다른 열로 예측하는 모델을 반복 | 가장 정교, 무거움(실험적 기능) |

---

# B. pandas 인덱싱 (loc / iloc)

DataFrame에서 원하는 행·열·값을 뽑아내는 핵심 문법이다.

## 6. pd06_1 — loc / iloc 기본

```python
df = pd.DataFrame(data=data, index=['031','059','033','045','023'],
                  columns=['종목명','시가','종가'])
```

### ★ 핵심 1: 대괄호 기본 인덱싱은 "열(컬럼)이 기준"

```python
# df['031']   → KeyError (인덱스로는 못 뽑음)
print(df['종목명'])          # OK: 컬럼명으로 '열'을 뽑는다
print(df['종목명']['045'])   # OK: 열 먼저 → 그 안에서 행 → '네이버'
# df['045']['종목명']  → 에러 (순서가 '열→행'이라 반대로는 안 됨)
```
- **pandas의 `df[...]`는 "열 → 행" 순서**다. `df['종목명']['045']`처럼 **컬럼을 먼저** 지정한다. 이게 헷갈림의 핵심이라 "판다스는 **열행**(열이 먼저)"으로 외운다.

### ★ 핵심 2: loc(라벨) vs iloc(정수 위치)

```python
# loc  : 인덱스 '라벨(이름)' 기준
# iloc : 행 '번호(정수 위치)' 기준  → "int loc"로 암기
print(df.loc['045'])   # 라벨 '045'인 행 (네이버)
print(df.iloc[3])      # 4번째(0부터 세서 3) 행 (네이버)
# df.iloc['045'] → 에러(정수 아님), df.loc[3] → 에러(라벨 아님)
```
- **`loc`는 라벨(이름)**, **`iloc`는 정수 위치(0,1,2…)**. `iloc`의 **i는 integer(정수)** → "**int loc**"로 외우면 안 헷갈린다.
- 여기 인덱스가 `'045'` 같은 **문자열**이라, 숫자 3으로 뽑으려면 반드시 `iloc`를 써야 한다.

### 특정 "값" 하나 뽑기 (행+열)

```python
print(df.iloc[3, 2])          # 3번행 2번열 = 1500 (iloc는 행,열 둘 다 정수)
print(df.iloc[3]['종가'])      # 행 먼저 뽑고 → 컬럼명으로
print(df.loc['045', '종가'])   # loc는 라벨,라벨 = 1500
print(df.loc['045']['종가'])   # 행 라벨 → 컬럼명
# df.iloc[3, '종가'] → 에러 (iloc에 컬럼명 불가)
# df.loc['045', 2]   → 에러 (loc에 정수 위치 불가)
```
- **`iloc[행, 열]`은 둘 다 정수**, **`loc[행, 열]`은 둘 다 라벨**로 써야 한다. 섞으면(iloc에 컬럼명, loc에 정수) 에러.
- `df.iloc[3].loc['종가']`처럼 **한 번 뽑은 뒤 다시 loc/iloc를 이어 붙일 수도** 있다(앞에서 행을 뽑아 Series가 되면, 그 Series에 또 인덱싱).

## 7. pd06_2 — loc / iloc 범위(슬라이싱)

여러 행을 **범위로** 뽑을 때.

```python
# 아모레(045)와 네이버(023)의 시가
print(df.iloc[3:5, 1])       # 3~4번 행, 1번 열(시가)  ★ iloc[행범위, 열]
print(df.iloc[3:5]["시가"])   # 3~4번 행을 뽑고 → '시가' 열
print(df.iloc[3:5].iloc[1])  # 3~4번 행 중 다시 1번째 행
# df.iloc[3:5][1] → 에러 (열 번호를 []로는 못 뽑음)
```
- **`iloc[3:5]`**: 3번부터 **5번 앞(=4번)까지** — 파이썬 슬라이싱처럼 **끝은 포함 안 한다.**
- **`iloc[행범위, 열]`** 형태로 행은 범위, 열은 번호를 함께 줄 수 있다. 단 `df.iloc[3:5][1]`처럼 **대괄호로 열 번호**를 주는 건 안 되고, `iloc[3:5, 1]`로 쉼표 안에 넣어야 한다.

---

## 한 줄 요약

> **결측치 처리**는 단순→정교 순으로: ①**삭제**(`dropna`, axis=0 행/1 열), ②**채우기**(`fillna` 평균·중위·0, `ffill`/`bfill` 앞·뒷값), ③**보간**(`interpolate` — 중간은 선형, 끝은 ffill, 맨 앞은 NaN), ④**모델 예측**, ⑤트리모델(결측에 강함). sklearn **Imputer**는 `fit_transform` 규격으로 파이프라인에 끼우며, **SimpleImputer**(열 통계값: mean·median·most_frequent·constant) → **KNNImputer**(가까운 이웃 K개로) → **IterativeImputer**(다른 열로 예측하는 모델을 반복, `estimator`에 XGBoost 등) 순으로 정교해진다. **pandas 인덱싱**은 `df[...]`가 **열 먼저**(열행)이고, **`loc`=라벨 / `iloc`=정수 위치**("int loc")이며 `iloc[행,열]`은 둘 다 정수·`loc[행,열]`은 둘 다 라벨로 쓰고, 범위는 `iloc[3:5, 1]`처럼 쉼표 안에 행범위·열을 넣는다(끝 번호는 미포함).
