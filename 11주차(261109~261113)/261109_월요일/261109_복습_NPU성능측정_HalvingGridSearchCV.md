# 261109 복습 — NPU LLM 성능 측정 지표 · HalvingGridSearchCV

> 앞부분은 NPU에 띄운 LLM의 **성능을 여러 지표로 측정**한다(지연시간·첫 응답까지 시간·처리량·토큰 수). 뒷부분은 하이퍼파라미터 튜닝의 세 번째 방식, **HalvingGridSearchCV**다.

## 목차
**A. NPU LLM 성능 측정**
1. kingkeras + f11 — 공통 모듈 & 기본 호출
2. f12 — 지연시간(Latency)
3. f13 — 스트리밍
4. f14 — TTFC (첫 조각까지 시간), 3가지 구현
5. f15 — Prefill / Decode
6. f16 — usage_metadata (토큰 수)
7. f17 — TPS (정확한 토큰 기반)
8. f18 — 프롬프트별 비교
9. 성능 지표 한눈 정리
**B. 하이퍼파라미터 튜닝**
10. m10 — HalvingGridSearchCV

---

# A. NPU LLM 성능 측정

## 1. kingkeras + f11 — 공통 모듈 & 기본 호출

어제처럼 LLM 생성을 모듈(`kingkeras.py`)로 빼두고 가져다 쓴다.

```python
# kingkeras.py
from langchain_openai import ChatOpenAI
def get_llm(max_tokens=None, temperature=None, top_p=None):
    return ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY",
                      max_tokens=max_tokens, temperature=temperature, top_p=top_p)
```
```python
# f11_basic.py — 가장 기본
from kingkeras import get_llm
llm = get_llm(max_tokens=1000)
response = llm.invoke("NPU와 GPU의 차이를 설명해줘.")
print(response.content)
```
- 이후 모든 파일이 `from kingkeras import get_llm`으로 LLM을 가져와 **성능 측정만** 다르게 한다.

## 2. f12 — 지연시간(Latency)

**지연시간(Latency)** 은 요청부터 응답 완료까지 걸린 **총 시간**이다. 한 번만 재면 들쭉날쭉하니 여러 번 재서 평균을 낸다.

```python
times = []
for i in range(5):                       # 5번 반복
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
average = sum(times) / len(times)        # 평균
print("평균:", round(average, 2), "초")
```
- 같은 요청을 5번 반복해 평균 지연시간을 구한다. 한 번 측정은 우연에 흔들리므로 **여러 번 평균**이 더 믿을 만하다.

## 3. f13 — 스트리밍

```python
for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)
```
- 지난 시간에 배운 스트리밍. 응답을 조각(chunk)씩 받아 실시간 출력한다. 다음 지표(TTFC)의 바탕이 된다.

## 4. f14 — TTFC (첫 조각까지 시간), 3가지 구현

**TTFC(Time To First Chunk)** 는 요청 후 **첫 번째 응답 조각이 나오기까지** 걸린 시간이다. 전체 응답 시간(Latency)과 달리, 사용자가 **"반응이 시작됐다"고 체감하는 속도**를 나타낸다(스트리밍에서 중요).

같은 측정을 점점 간결하게 세 번 구현했다.

```python
# f14_1 — 플래그로 '첫 조각일 때만' 시간 기록 (출력도 함께)
start = time.perf_counter(); first_time = None
for chunk in llm.stream(prompt):
    if chunk.content:
        if first_time is None:                      # 아직 기록 전이면
            first_time = time.perf_counter() - start  # 첫 조각 시간 기록
        print(chunk.content, end="", flush=True)
```
```python
# f14_2 — 첫 조각 받으면 바로 break (측정만, 출력 안 함)
for chunk in llm.stream(prompt):
    if chunk.content:
        first_time = time.perf_counter() - start
        break                                        # 더 볼 필요 없으니 중단
```
```python
# f14_3 — next(iter(...))로 첫 조각만 꺼내기 (가장 간결)
chunk = next(iter(llm.stream(prompt)))               # 스트림의 첫 요소 하나만
first_time = time.perf_counter() - start
```

- 세 코드 모두 **"첫 조각까지의 시간"** 이라는 같은 값을 잰다. 방식만 다르다:
  - `f14_1`: `first_time is None` 플래그로 첫 조각일 때 한 번만 기록(전체 출력은 계속).
  - `f14_2`: 첫 조각을 받자마자 `break`로 반복 중단(측정 전용).
  - `f14_3`: **`next(iter(...))`** 로 반복자의 첫 요소 하나만 꺼낸다 — 가장 짧다.
- **`next(iter(...))`**: `iter()`로 반복자를 만들고 `next()`로 그 첫 값만 꺼내는 파이썬 관용구. for문 없이 "맨 앞 하나"만 얻을 때 쓴다.

## 5. f15 — Prefill / Decode

LLM 추론은 **두 단계**로 나뉜다. 프롬프트 길이·출력 길이를 다르게 해 응답시간을 비교하며 이를 관찰한다.

```python
prompts = [
    "NPU란?",                                           # 짧음
    "NPU와 GPU의 차이를 10가지 설명해줘.",               # 중간
    "NPU의 구조와 장점, 활용 분야를 더더더 자세히 설명해줘." # 긺(출력 많음)
]
for prompt in prompts:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    print("시간:", round(time.perf_counter() - start, 2), "초")
```

- **Prefill(프리필)**: 입력 프롬프트 전체를 **한 번에 읽어 처리**하는 단계(병렬 처리라 비교적 빠름).
- **Decode(디코드)**: 출력 토큰을 **하나씩 순서대로 생성**하는 단계(토큰 수만큼 반복이라 느림).
- 그래서 **출력이 길어질수록(decode가 많아질수록) 응답시간이 크게 늘어난다.** 세 프롬프트의 시간 차이로 이 경향을 확인한다. (입력이 매우 길면 prefill 시간도 늘지만, 보통 decode가 전체 시간을 좌우한다.)

## 6. f16 — usage_metadata (토큰 수)

응답 객체에는 **실제로 쓰인 토큰 수** 정보가 들어 있다.

```python
response = llm.invoke("NPU란 무엇인가")
print(response.usage_metadata)
# {'input_tokens': ..., 'output_tokens': ..., 'total_tokens': ...}
```
- **`response.usage_metadata`**: 입력 토큰(`input_tokens`)·출력 토큰(`output_tokens`)·합계를 알려준다. 지난 시간엔 `max_tokens`(설정한 상한)로 TPS를 어림했지만, 여기서 **실제 출력 토큰 수**를 얻어 더 정확히 계산할 수 있다.

## 7. f17 — TPS (정확한 토큰 기반)

**TPS(Tokens Per Second)** = 초당 생성 토큰 수. f16의 실제 출력 토큰으로 정확히 계산한다.

```python
start = time.perf_counter()
response = llm.invoke(prompt)
elapsed = time.perf_counter() - start

tokens = response.usage_metadata["output_tokens"]   # 실제 출력 토큰 수
tps = tokens / elapsed
print("TPS:", round(tps, 2))
# 출력 토큰: 500 / 전체 시간: 5.4초 / TPS: 92.55
```
- 지난 시간(`max_tokens`로 나눔)과 달리, **`usage_metadata["output_tokens"]`(실제 생성된 토큰)** 로 나눠 정확한 처리량을 얻는다. (상한을 다 채우지 않고 끝나는 경우도 있어 실제 토큰이 더 정확하다.)

## 8. f18 — 프롬프트별 비교

여러 프롬프트의 **입력·출력 토큰과 응답시간**을 한꺼번에 비교한다.

```python
for prompt in prompts:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    print("입력 토큰:", response.usage_metadata["input_tokens"])
    print("출력 토큰:", response.usage_metadata["output_tokens"])
    print("응답시간:", round(elapsed, 2), "초")
```
- 프롬프트마다 입력·출력 토큰 수와 시간을 뽑아, **출력 토큰이 많을수록 시간이 느는** 관계(5번 decode)를 수치로 확인한다.

## 9. 성능 지표 한눈 정리

| 지표 | 뜻 | 측정 방법 | 무엇을 보나 |
|---|---|---|---|
| **Latency(지연시간)** | 요청~응답 완료 총 시간 | start/end 차이 (여러 번 평균) | 전체 응답 속도 |
| **TTFC** | 첫 조각까지 시간 | 스트림 첫 chunk 시각 | 체감 반응 시작 속도 |
| **TPS** | 초당 출력 토큰 수 | 출력토큰 ÷ 시간 | 생성 처리량 |
| **usage_metadata** | 입력·출력 토큰 수 | `response.usage_metadata` | TPS 등의 정확한 근거 |
| **Prefill/Decode** | 입력 처리 / 출력 생성 단계 | 프롬프트·출력 길이별 시간 | 시간이 어디서 드는지 |

---

# B. 하이퍼파라미터 튜닝

## 10. m10 — HalvingGridSearchCV

튜닝 방식의 세 번째. 지금까지 **GridSearch**(모든 조합 전부)·**RandomSearch**(무작위 일부)를 배웠다. **HalvingGridSearchCV**는 **"연속 절반 탈락(successive halving)"** 전략으로, 토너먼트처럼 유망한 후보에 자원을 점점 몰아준다.

```python
from sklearn.experimental import enable_halving_search_cv   # (실험적 기능 활성화)
from sklearn.model_selection import HalvingGridSearchCV

model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
                    refit=True,
                    min_resources=50,     # 1라운드에 쓸 자원(데이터 수) 시작값
                    max_resources=1000,   # 최대 자원
                    factor=2,             # 라운드마다 자원 ×2, 후보는 ÷2
                    verbose=1)
model.fit(x_train, y_train)
print('model.score :', model.score(x_test, y_test))
print('best_score :', model.best_score_)
```

**작동 방식 (토너먼트)**:
```
1라운드: 적은 자원(데이터 50개)으로 → 많은 후보 전부 평가 → 성적 하위 절반 탈락
2라운드: 자원 2배(100개)로 → 살아남은 후보만 평가 → 또 하위 절반 탈락
3라운드: 자원 2배(200개)로 → ...                 → 반복
마지막 : 소수 후보를 많은 자원으로 정밀 평가 → 최종 승자
```

- **핵심 아이디어**: 처음부터 모든 후보에 전체 데이터를 쓰면 느리다. 그래서 **적은 데이터로 가망 없는 후보를 빨리 걸러내고**, 살아남은 유망 후보에만 점점 더 많은 자원을 투입한다. GridSearch보다 빠르게 좋은 조합을 찾는다.
- **파라미터**:
  - `min_resources`: 첫 라운드에 쓸 자원(보통 데이터 샘플 수)의 시작값.
  - `max_resources`: 마지막에 쓸 최대 자원(전체 데이터 수 근처).
  - `factor`: 라운드마다 **자원을 몇 배 늘리고 후보를 얼마로 줄일지**. `factor=2`면 자원 2배·후보 1/2.
- **주의(보충)**: `HalvingGridSearchCV`는 사이킷런의 **실험적(experimental) 기능**이라, 보통 `from sklearn.experimental import enable_halving_search_cv`를 **먼저 import해야** 쓸 수 있다. (코드에서 그 줄을 주석 처리하고 내부 경로로 직접 가져왔는데, 버전에 따라 동작이 갈릴 수 있어 공식 방식은 experimental을 활성화하는 것이다.)

**튜닝 3방식 비교**

| 방식 | 전략 | 속도 |
|---|---|---|
| GridSearchCV | 모든 조합 × 전체 데이터 | 느림 |
| RandomizedSearchCV | 무작위 일부 조합 | 빠름 |
| HalvingGridSearchCV | 적은 자원→유망 후보에 자원 집중(절반씩 탈락) | 빠름(체계적) |

---

## 한 줄 요약

> **NPU LLM 성능 측정**: **Latency**(요청~완료 총 시간, 여러 번 평균), **TTFC**(첫 조각까지 시간 = 체감 반응 속도, `next(iter(stream))`로 간결 측정), **TPS**(초당 출력 토큰 = `usage_metadata["output_tokens"] ÷ 시간`), **usage_metadata**(입력·출력 토큰 수)로 모델 속도를 여러 각도에서 잰다. 추론은 **Prefill**(입력 한 번에 처리)·**Decode**(출력 토큰 하나씩 생성) 두 단계라, 출력이 길수록 느리다. **HalvingGridSearchCV**는 적은 자원으로 많은 후보를 걸러내고 유망 후보에 자원을 2배씩 몰아주는(절반씩 탈락) 방식으로, Grid(전부)·Random(무작위 일부)에 이은 세 번째 튜닝 전략이다(`min_resources`·`max_resources`·`factor`, experimental 활성화 필요).
