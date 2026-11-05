# 261105 복습 — NPU LLM 심화(LangChain·스트리밍·모듈화) · 파이썬 모듈/패키지 · GridSearchCV

> 오늘은 세 묶음이다. ① NPU에 띄운 LLM을 **LangChain으로 연결하고, 스트리밍·처리량(TPS)·모듈화**까지 다룬다. ② 그 과정에서 쓰인 **파이썬 모듈·패키지와 import** 문법을 정리한다. ③ 머신러닝에서 **하이퍼파라미터를 자동으로 튜닝하는 GridSearchCV**를 배운다.

## 목차
**A. NPU LLM 심화**
1. f05 — LangChain으로 NPU 모델 연결
2. f06 — 처리량(TPS) 측정 & 생성 파라미터
3. f07 — 스트리밍 출력
4. f08 + a_kingkeras — LLM 생성 모듈화
**B. 파이썬 모듈과 패키지**
5. 모듈 import (p01~p03)
6. `__name__` 과 `if __name__=='__main__'` (p11~p23)
7. 변수 스코프와 from-import (p31~p33)
8. 패키지(폴더) import (machine/, p41~p51)
**C. 하이퍼파라미터 튜닝**
9. m06 — 여러 데이터 × 모든 모델 비교
10. m07 — for문으로 수동 튜닝
11. m08 — GridSearchCV 자동 튜닝

---

# A. NPU LLM 심화

## 1. f05 — LangChain으로 NPU 모델 연결

어제까지는 OpenAI SDK(`openai` 패키지)로 NPU 서버에 붙었다. 오늘은 **LangChain의 `ChatOpenAI`** 로 붙는다. (지난 RAG 때 쓰던 그 클래스 그대로, 주소만 NPU 로컬 서버로)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",   # 내 NPU 서버
    api_key="EMPTY",
)

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘.")
print(response.content)
```

- OpenAI SDK의 `client.chat.completions.create(...)` 대신, LangChain은 **`llm.invoke("질문")`** 한 줄이면 된다. 더 간결하다.
- **핵심**: NPU 서버가 OpenAI 호환이라, OpenAI SDK든 LangChain이든 **`base_url`만 NPU로 바꿔** 똑같이 쓸 수 있다. LangChain을 쓰면 이후 체인·RAG·에이전트에 바로 연결하기 좋다.

## 2. f06 — 처리량(TPS) 측정 & 생성 파라미터

NPU가 초당 토큰을 얼마나 뽑는지(**TPS, tokens/second**) 측정한다.

```python
import time

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1", api_key="EMPTY",
    max_tokens=1000,      # 최대 출력 토큰 수
    temperature=2,        # 창의성(무작위성) — 높을수록 다양/엉뚱
    top_p=0.5,            # 누적 확률 샘플링 범위
)

start = time.perf_counter()
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘.")
end = time.perf_counter()
elapsed = round(end - start, 2)

print("추론시간:", elapsed, "초")
print("평균 처리량:", round(1000/elapsed, 2), "tokens/s")
# 추론시간: 6.92 초 / 평균 처리량: 144.51 tokens/s
```

- **TPS = 출력 토큰 수 ÷ 걸린 시간**. 모델·하드웨어의 추론 속도를 나타내는 지표다(여기선 약 144 tokens/s).
- **`time.perf_counter()`**: 정밀한 시간 측정용. 시작·끝을 재서 차이로 소요 시간을 구한다.
- **생성 파라미터**:
  - `max_tokens`: 최대 출력 길이.
  - `temperature`: 창의성/무작위성. 0이면 매번 같은(일관된) 답, 높을수록 다양하고 엉뚱해진다(2는 꽤 높은 값).
  - `top_p`: 다음 토큰을 고를 때 확률 상위 몇 %만 후보로 둘지(0.5면 상위 50% 확률 질량에서 샘플링). temperature와 함께 답의 다양성을 조절한다.

## 3. f07 — 스트리밍 출력

답변이 **다 완성될 때까지 기다리지 않고, 생성되는 대로 조각(토큰)씩 실시간 출력**하는 방식이다. (ChatGPT가 글자를 타이핑하듯 보여주는 그 효과)

```python
# f07_stream01 — OpenAI SDK
streaming = client.chat.completions.create(
    model=model, messages=[...], max_tokens=300,
    stream=True,                      # ★ 스트리밍 켜기
)
for chunk in streaming:
    if chunk.choices:
        content = chunk.choices[0].delta.content   # 이번 조각의 글자
        if content:
            print(content, end="", flush=True)     # 줄바꿈 없이 즉시 출력
```
```python
# f07_stream02_lc — LangChain (훨씬 간결)
for chunk in llm.stream("NPU와 GPU의 차이점을 한국어로 설명해줘"):
    print(chunk.content, end="", flush=True)
```

- **`stream=True`**(SDK) / **`llm.stream(...)`**(LangChain): 응답을 한 번에 받지 않고 **조각(chunk)으로 나눠** 받는다.
- **`end="", flush=True`**: `end=""`는 조각마다 줄바꿈 없이 이어 붙이고, `flush=True`는 버퍼에 쌓아두지 않고 **즉시 화면에 출력**한다(실시간 효과의 핵심).
- SDK는 `chunk.choices[0].delta.content`로, LangChain은 `chunk.content`로 조각의 내용을 꺼낸다. LangChain이 코드가 훨씬 짧다.
- **왜 쓰나**: 긴 답변을 다 기다리지 않고 바로바로 보여줘 **체감 반응 속도**를 높인다.

## 4. f08 + a_kingkeras — LLM 생성 모듈화

매번 반복되는 `ChatOpenAI(...)` 설정을 **함수로 빼서 모듈(파일)로** 만들고, 다른 파일에서 가져다 쓴다.

```python
# a_kingkeras.py — 공통 LLM 생성 함수를 모듈로
from langchain_openai import ChatOpenAI

def get_llm(max_tokens=1000, temperature=0, top_p=0.9):
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1", api_key="EMPTY",
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
    )
    return llm
```
```python
# f08_pkg01_invoke.py — 모듈에서 가져와 사용
from a_kingkeras import get_llm

llm = get_llm(max_tokens=1000, temperature=0.9, top_p=1.0)
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘.")
print(response.content)
```
```python
# f08_pkg02_stream.py — 같은 모듈로 스트리밍
from a_kingkeras import get_llm
llm = get_llm()
for chunk in llm.stream("..."):
    print(chunk.content, end="", flush=True)
```

- **`get_llm()`** 하나로 LLM 설정을 한 곳에 모았다. 여러 파일에서 `from a_kingkeras import get_llm`으로 **재사용**하니, 설정이 바뀌어도 모듈 한 곳만 고치면 된다.
- 이렇게 "공통 코드를 별도 파일로 빼서 import해 쓰는" 흐름이 자연스럽게 **파이썬 모듈/패키지 개념(B)** 으로 이어진다.

---

# B. 파이썬 모듈과 패키지

**모듈 = 함수·변수를 담은 하나의 `.py` 파일**, **패키지 = 모듈들을 담은 폴더**다. 다른 파일에서 `import`로 가져와 재사용한다.

## 5. 모듈 import (p01~p03)

```python
# p01_car.py
def drive():
    print("운전하다")
drive()            # ← 파일 맨 아래에서 바로 호출

# p03_do.py
import python.p01_car as p01_car    # 다른 파일을 모듈로 가져오기
import python.p02_tv as p02_tv
p01_car.drive()                     # 모듈명.함수명() 으로 호출
p02_tv.watch()
```

- **`import <모듈> as <별칭>`**: 다른 `.py`를 가져와 `별칭.함수()`로 쓴다.
- **주의(문제점)**: `p01_car.py`는 파일 맨 아래에 `drive()` 호출이 **그대로 있다.** 그래서 `import`하는 **그 순간에도** "운전하다"가 실행돼 버린다(원치 않는 자동 실행). 이 문제를 6번에서 해결한다.

## 6. `__name__` 과 `if __name__=='__main__'` (p11~p23)

```python
# p11_car.py
def drive(): print("운전하다")
drive()
print("p11_car.py의 module 이름은 :", __name__)
```

- **`__name__`**: 파이썬이 각 파일에 자동으로 넣어주는 "이 파일의 이름" 변수다.
  - 그 파일을 **직접 실행**하면 → `__name__` 은 `'__main__'`
  - 다른 파일에서 **import되면** → `__name__` 은 그 파일의 모듈명(`'python.p11_car'` 등)

```python
# p21_car.py
def drive(): print("운전하다")

if __name__=='__main__':   # ★ 직접 실행할 때만 아래가 돈다
    drive()
```

- **`if __name__=='__main__':`** 는 "이 파일을 **직접 실행**할 때만 이 블록을 돌려라"는 뜻이다. 다른 파일에서 **import하면 이 블록은 안 돈다.**
- **효과**: 5번의 문제(import만 해도 함수가 실행되던 것)를 막는다. `p23_main_do.py`에서 `import python.p21_car` 해도 `drive()`가 자동 실행되지 않고, `p21_car.drive()`로 **직접 호출할 때만** 실행된다.
- **왜 중요**: 모듈을 "가져다 쓰는 용도"와 "직접 돌리는 용도(테스트 등)"를 분리하는 파이썬의 핵심 관용구다. 거의 모든 실무 코드에 들어간다.

## 7. 변수 스코프와 from-import (p31~p33)

```python
# p31_sample.py
x = 111
def test(): print('x :', x)

# p32_main_func.py
import python.p31_sample as p31_sample
x = 222
def main_func(): print('x :', x)
main_func()           # x : 222  (이 파일의 x)
p31_sample.test()     # x : 111  (p31 모듈 안의 x)
```

- **각 모듈은 독립된 이름 공간(스코프)** 을 가진다. `p32`의 `x=222`와 `p31`의 `x=111`은 서로 다른 변수다. `test()`는 자기 모듈(p31)의 `x`를 본다.

```python
# p33_from.py — 두 가지 import 방식 비교
import python.p31_sample as p31_sample      # (1) 모듈 통째로
from python.p31_sample import test          # (2) 특정 함수만 콕 집어서

p31_sample.test()    # (1) 방식: 모듈명.함수()
test()               # (2) 방식: 함수명 바로
```

- **`import 모듈`** → `모듈.함수()`로 호출. **`from 모듈 import 함수`** → `함수()`로 바로 호출. 둘 다 같은 함수를 부르는 다른 문법이다.

## 8. 패키지(폴더) import (machine/, p41~p51)

**패키지 = 모듈(.py)들을 담은 폴더.** `machine/` 폴더 안에 `car.py`, `tv.py`가 있고, 그 안에 또 `test/` 폴더가 있다.

```python
# p41_import_module.py — 폴더.파일 전체 경로
import machine.car
import machine.tv
machine.car.drive()
machine.tv.watch()

# p42_from_module.py — from 으로 간결하게
from machine.car import drive
from machine.tv import watch
drive(); watch()

from machine import car, tv       # 폴더에서 모듈들을 가져오기
car.drive(); tv.watch()

# p51_from_module.py — 하위 폴더(test)까지
from machine.test import test_car, test_tv
test_car.drive(); test_tv.watch()
```

- **`import machine.car`**: `machine` 폴더의 `car` 모듈을 가져온다(`폴더.파일`).
- **`from machine import car, tv`**: 폴더에서 여러 모듈을 한 번에.
- **`from machine.test import ...`**: 하위 폴더(`machine/test/`)도 점(`.`)으로 이어 접근한다.
- 폴더 구조(`python/` ↔ `machine/`)에 따라 import 경로가 달라진다는 것이 핵심이다.

---

# C. 하이퍼파라미터 튜닝

## 9. m06 — 여러 데이터 × 모든 모델 비교

어제 `all_estimators`를 **여러 데이터셋에 한 번에** 적용하고, `StratifiedKFold`로 교차검증해 **데이터별 최고 모델**을 찾는다.

```python
datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

all = all_estimators(type_filter='classifier')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)   # 분류 → Stratified

for index, value in enumerate(datasets):      # 데이터셋마다
    x, y = value
    # ...train_test_split, 스케일링...
    for name, model in all:                   # 모든 모델마다
        try:
            scores = cross_val_score(model(), x_train, y_train, cv=kfold)
            y_pred = cross_val_predict(model(), x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_pred)
            # 이름·점수 모아두기
        except: continue                      # 오류 모델은 건너뜀
    # 가장 높은 acc의 모델 출력 (np.argmax)
```

- **이중 for문**: 바깥은 데이터셋(4개), 안쪽은 모든 분류 모델. 데이터마다 전 모델을 교차검증해 **`np.argmax`로 최고 성능 모델**을 뽑는다.
- 분류라서 어제 배운 **`StratifiedKFold`**(클래스 비율 유지)를 쓴다. `try/except: continue`로 에러 모델은 조용히 건너뛴다.

## 10. m07 — for문으로 수동 튜닝

모델의 **하이퍼파라미터(사람이 정하는 설정값)** 를 바꿔가며 최고 조합을 찾는다. 먼저 **for문으로 직접(수동)** 해 본다.

```python
learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]   # 5개
max_depth = [3,4,5,6,7]                            # 5개
best_score = 0; best_parameters = ''

for lr in learning_rate:            # 5 ×
    for md in max_depth:            #   5 = 25가지 조합
        model = HistGradientBoostingClassifier(learning_rate=lr, max_depth=md)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        if score > best_score:      # 더 좋으면 갱신
            best_score = score
            best_parameters = lr, md

print("최고 점수 :", best_score, "/ 최적 매개변수 :", best_parameters)
```

- **하이퍼파라미터**: 학습률(`learning_rate`)·트리 깊이(`max_depth`)처럼 **학습 전에 사람이 정하는 값**. 값에 따라 성능이 크게 달라진다.
- **이중 for로 모든 조합(5×5=25가지)을 전부 시도**하고, 점수가 갱신될 때마다 최고 조합을 기록한다. 직관적이지만 **코드로 일일이 짜야 하고 교차검증도 안 들어간** 방식이다.

## 11. m08 — GridSearchCV 자동 튜닝

10번의 "모든 조합 수동 탐색 + 교차검증"을 **한 번에 자동화**하는 것이 `GridSearchCV`다.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)

parameters = [                                       # 시도할 조합들
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1,0.01,0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1,0.01,0.001]},                           # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1,0.01,0.001]},                     # 12
]  # 총 42가지

model = GridSearchCV(XGBClassifier(), parameters, cv=kfold,
                     verbose=1, n_jobs=-1)           # 격자탐색 + 교차검증
model.fit(x_train, y_train)

print("최적의 파라미터 :", model.best_params_)       # 가장 좋았던 조합
print("best_score :", model.best_score_)             # 그 조합의 교차검증 평균 점수
print("model.score :", model.score(x_test, y_test))  # test로 최종 평가
```

- **GridSearchCV = 격자(Grid) 탐색 + 교차검증(CV)**. `parameters`에 적은 **모든 조합을 격자처럼 전부 시도**하되, 각 조합을 **K-Fold 교차검증**으로 평가해 가장 좋은 조합을 자동으로 찾는다. (m07의 수동 for + 교차검증을 합쳐 자동화한 것)
- **`parameters`**: 딕셔너리의 리스트로 조합을 정의한다. 각 딕셔너리 안 값들의 곱이 조합 수(18+12+12=42가지).
- **연산량 주의**: 42개 조합 × 5-fold = **210번 학습**한다. 체계적이지만 느리다. `n_jobs=-1`(모든 CPU 코어 사용)·`verbose=1`(진행 표시)로 돌린다.
- **결과 꺼내기**: `best_params_`(최적 조합), `best_estimator_`(그 조합의 모델), `best_score_`(교차검증 평균 점수), `score(x_test, y_test)`(남겨둔 test로 최종 확인).

---

## 한 줄 요약

> **NPU LLM**: OpenAI SDK 대신 **LangChain `ChatOpenAI`** 로도 NPU 서버에 붙을 수 있고(`base_url`만 로컬), `llm.invoke`/`llm.stream`으로 추론·스트리밍(조각씩 실시간 출력, `end=""`·`flush=True`)한다. 처리량은 **TPS = 출력토큰/시간**으로 재고, 반복되는 설정은 `get_llm()` 함수로 **모듈화**해 import 재사용한다. **파이썬 모듈/패키지**: `.py`는 모듈·폴더는 패키지이고 `import`로 재사용하는데, 파일 끝의 호출은 import 시에도 실행되므로 **`if __name__=='__main__':`** 으로 직접 실행 때만 돌게 막는다(핵심 관용구). `import 모듈` vs `from 모듈 import 함수`, 폴더는 `import 패키지.모듈`·`from 패키지 import 모듈`로 접근한다. **하이퍼파라미터 튜닝**: 수동 for문(m07)으로 모든 조합을 돌려 best를 찾던 것을, **`GridSearchCV`(격자 탐색 + 교차검증)** 로 자동화해 `best_params_`·`best_score_`를 얻고 test로 최종 평가한다(조합 수 × fold 수만큼 학습하므로 느리지만 체계적).
