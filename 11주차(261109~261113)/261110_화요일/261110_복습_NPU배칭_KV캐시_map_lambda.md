# 261110 복습 — NPU 추론 최적화(Continuous Batching·KV Cache) · 파이썬 map/lambda

> 앞부분은 NPU가 LLM 추론을 빠르게 하는 두 가지 핵심 기법, **연속 배칭(Continuous Batching)** 과 **KV 캐시(KV Cache)** 를 직접 측정하며 이해한다. 뒷부분은 파이썬의 **map·lambda** 문법이다.

## 목차
**A. NPU 추론 최적화**
1. kingkeras 수정 — run_once / get_cache_metrics
2. f21 — Continuous Batching (동시 요청)
3. f22 — KV Cache (캐시 사용률 관찰)
**B. 파이썬 map / lambda**
4. p01 — map
5. p02 — lambda

---

# A. NPU 추론 최적화

## 1. kingkeras 수정 — 측정 도우미 함수 추가

공통 모듈에 측정용 함수 두 개를 더했다.

```python
def run_once(promt, max_tokens=1000):          # 한 번 호출하고 결과를 묶어 반환
    llm = get_llm(max_tokens)
    start = time.perf_counter()
    response = llm.invoke(promt)
    elapsed = time.perf_counter() - start
    usage = response.usage_metadata or {}
    return {
        "elapsed": elapsed,
        "input_tokens": usage.get("input_tokens", 0),    # 없으면 0 (안전하게)
        "output_tokens": usage.get("output_tokens", 0),
        "text": response.content,
    }
```

- **`run_once()`**: 한 번 추론하고 **시간·입력토큰·출력토큰·답변**을 딕셔너리로 묶어 돌려준다. 반복 측정 코드를 깔끔하게 하려고 만든 도우미다.
- **`usage.get("input_tokens", 0)`**: 딕셔너리에서 값을 꺼내되 **키가 없으면 0**을 준다(`usage["input_tokens"]`는 키가 없으면 에러). `response.usage_metadata or {}`도 "값이 없으면 빈 딕셔너리"로 대비한 안전장치다.

```python
import urllib.request
def get_cache_metrics():                       # 서버의 캐시 지표 읽기
    url = "http://127.0.0.1:8000/metrics"
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read().decode("utf-8")
    # ... 아래 지표를 줄 단위로 파싱 ...
    #   furiosa_llm_prefix_cache_hits    : 프리픽스 캐시 적중 횟수
    #   furiosa_llm_prefix_cache_queries : 프리픽스 캐시 조회 횟수
    #   furiosa_llm_kv_cache_usage_percent : KV 캐시 사용률
    # 적중률 = hits / queries * 100
```

- **`get_cache_metrics()`**: NPU 서버가 제공하는 **`/metrics` 주소**에 접속해 캐시 상태 지표를 읽어온다. `urllib.request`로 그 페이지 텍스트를 받아 줄 단위로 훑으며 원하는 숫자(프리픽스 캐시 적중/조회, KV 캐시 사용률)를 뽑는다. 서빙 서버가 성능 지표를 이 주소로 공개한다.

## 2. f21 — Continuous Batching (동시 요청)

**연속 배칭(Continuous Batching)** 은 **여러 요청을 동시에 받아 NPU가 묶어서 처리**하는 기법이다. NPU/GPU는 병렬 연산에 강해, 요청을 하나씩 따로 처리하는 것보다 **여러 개를 한 번에 묶으면 전체 처리량(TPS)이 올라간다.** 동시 요청 수를 늘려가며 처리량 변화를 관찰한다.

```python
from concurrent.futures import ThreadPoolExecutor
from kingkeras import run_once

prompts = [f"NPU 추론 최적화 방법 {i}번을 자세히 설명해줘." for i in range(8)]   # 질문 8개

def ask(prompt):
    return run_once(prompt, 1000)

for workers in [1, 2, 4, 8]:                 # 동시 요청 수를 1→2→4→8로
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, prompts))     # 여러 요청을 동시에
    elapsed = time.perf_counter() - start
    total_tokens = sum(r['output_tokens'] for r in results)
    print(f"동시 요청 수: {workers} / 전체 시간: {elapsed:.2f}초 / 처리량: {total_tokens/elapsed:.2f} TPS")
```

- **`ThreadPoolExecutor(max_workers=N)`**: 스레드 N개로 작업을 **동시에** 돌리는 파이썬 도구. **`pool.map(함수, 리스트)`** 는 리스트의 각 항목에 함수를 병렬로 적용한다. `max_workers`가 동시 요청 수다.
- **관찰 포인트**: 동시 요청 수(1→2→4→8)를 늘리면, 서버가 요청을 배칭해 처리하므로 **전체 처리량(TPS)이 늘어나는** 경향을 본다(하드웨어 한계까지). 요청을 하나씩 순차 처리할 때보다 효율적이다.
- **`f21_2`** 는 `max_tokens=50`으로 줄이고 `lambda prompts: run_once(prompts, 50)`로 넘긴 버전이다(출력 토큰 수를 바꿔 비교).
- (참고: 이 파일들은 `print`와 `avg_latency` 계산이 for 루프 밖에 있어 **마지막 `workers=8` 결과만** 출력되고, "평균 요청 지연" 출력은 값이 비어 있다. 측정 개념 이해가 목적이라 출력 위치는 넘어가도 된다.)

## 3. f22 — KV Cache (캐시 사용률 관찰)

**KV 캐시(Key-Value Cache)** 는 LLM이 토큰을 하나씩 생성할 때, **앞서 계산한 토큰들의 중간 결과(Key·Value)를 저장해 재사용**하는 장치다. 이게 없으면 새 토큰마다 지금까지의 모든 토큰을 처음부터 다시 계산해야 해 매우 느리다. **입력·출력이 길수록 저장할 게 많아져 KV 캐시 사용량(메모리)이 늘어난다.** 입력/출력 길이를 바꿔가며 이를 관찰한다.

```python
# f22_kv_cache1 — 길이별로 토큰 수·시간 비교
short_prompt = "NPU란 무엇인가?"
long_prompt = ("NPU는 인공지능 연산을 가속하는 프로세서다." * 100) + "위 내용을 요약해줘."

tests = [
    ("짧은 입력", short_prompt, 80),
    ("긴 입력", long_prompt, 80),
    ("긴 입력과 긴 출력", long_prompt, 400),
]
for name, prompt, max_tokens in tests:
    result = run_once(prompt, max_tokens)
    print("입력 토큰:", result["input_tokens"], "출력 토큰:", result["output_tokens"],
          "소요 시간:", round(result["elapsed"], 2))
    input("서버 KV Cache 사용률을 확인하고 싶다 Enter...")   # 잠깐 멈춤
```
```python
# f22_kv_cache2 — 매 실험 후 캐시 지표까지 출력
from kingkeras import run_once, get_cache_metrics
prompts = ["NPU란 무엇인가?", "NPU의 특징을 설명해줘."*100, "NPU의 특징을 설명해줘."*300]
for i, prompt in enumerate(prompts, 1):
    result = run_once(prompt, 300)
    print("입력 토큰:", result["input_tokens"], "응답 시간:", round(result["elapsed"],2))
    get_cache_metrics()     # 프리픽스 캐시 적중률 · KV 캐시 사용률 출력
```

- **입력/출력을 길게 할수록**(프롬프트를 `*100`, `*300`으로 반복) 입력 토큰과 KV 캐시 사용률이 올라가는 것을 본다. 긴 출력(decode가 많음)도 시간을 크게 늘린다.
- **`input("... Enter...")`**: 프로그램을 그 줄에서 **잠깐 멈춰** 사용자가 서버의 캐시 상태를 확인할 시간을 준다(엔터를 치면 계속).
- **프리픽스 캐시(Prefix Cache)**: 여러 요청이 **같은 앞부분(접두사)** 을 공유하면 그 부분 계산을 재사용한다. `get_cache_metrics()`의 **적중률(hit rate)** 이 높을수록 재사용이 잘 돼 효율적이다.
- **정리**: KV 캐시·프리픽스 캐시는 "이미 계산한 걸 다시 안 하려고 저장해 두는" 최적화다. 덕분에 빨라지지만 **메모리를 쓰므로**, 입력·출력이 길면 캐시 사용률(메모리)이 올라간다.

---

# B. 파이썬 map / lambda

데이터를 간결하게 처리하는 두 문법. 전처리·함수형 코드에서 자주 쓴다.

## 4. p01 — map

**`map(함수, 반복가능객체)`** 는 리스트 등의 **각 원소에 함수를 한 번씩 적용**한다.

```python
def multiply(x):
    return x * 10

numbers = [1, 2, 3, 4]
result = map(multiply, numbers)

print(result)        # <map object at 0x...>   ← 아직 계산 전(지연 평가)
print(list(result))  # [10, 20, 30, 40]        ← list()로 변환해야 값이 나옴
```

- `map`은 각 원소에 함수를 적용한 **결과를 바로 리스트로 주지 않고**, "필요할 때 꺼내 쓰는" map 객체(반복자)를 돌려준다. 그래서 `print(result)`는 주소만 보이고, **`list(result)`** 로 감싸야 실제 값이 나온다.
- for문으로 하나씩 돌며 처리하던 걸 한 줄로 줄인다.

## 5. p02 — lambda

**`lambda`** 는 이름 없이 한 줄로 만드는 **익명 함수**다.

```python
# p02_lambda1 — def 와 lambda 비교
def multiply(x):
    return x * 2
print(multiply(10))        # 20

aaa = lambda x : x * 2     # 위 함수의 한 줄 버전
print(aaa(10))             # 20
```
```python
# p02_lambda2 — 인자 여러 개
aaa = lambda x, y : x + y
print(aaa(1, 2))   # 3
print(aaa(3, 4))   # 7
```
```python
# p02_lambda3 — map + lambda + 조건식
score = [85, 45, 70, 55, 95]
results = list(map(lambda x: "합격" if x >= 60 else "불합격", score))
print(results)   # ['합격', '불합격', '합격', '불합격', '합격']
```

- **`lambda 인자 : 식`**: `def`로 함수를 따로 정의하지 않고 **한 줄로** 간단한 함수를 만든다. `aaa = lambda x: x*2`는 `def aaa(x): return x*2`와 같다.
- **`map`과 함께** 쓰면 강력하다. `p02_lambda3`은 점수 리스트의 각 값에 `lambda x: "합격" if x>=60 else "불합격"`(60 이상이면 합격, 아니면 불합격)을 적용해 한 줄로 합격 판정 리스트를 만든다.
- **`"합격" if x >= 60 else "불합격"`**: 파이썬의 **조건식(삼항 연산자)** — "조건이 참이면 앞, 거짓이면 뒤" 값을 돌려준다. `map`·`lambda`와 묶여 간결한 데이터 변환에 자주 쓰인다.

---

## 한 줄 요약

> **NPU 추론 최적화**: **Continuous Batching**은 여러 요청을 동시에 받아 NPU가 묶어 처리해 전체 처리량(TPS)을 높이는 기법으로, `ThreadPoolExecutor`로 동시 요청 수(1·2·4·8)를 늘려가며 확인한다. **KV Cache**는 이미 계산한 토큰들의 Key·Value를 저장해 재사용하는 장치로, 입력·출력이 길수록 사용량(메모리)이 늘며, 같은 접두사를 재사용하는 **프리픽스 캐시**의 적중률이 높을수록 효율적이다(서버 `/metrics`에서 지표 확인). **파이썬**: `map(함수, 리스트)`는 각 원소에 함수를 적용(결과는 `list()`로 변환), `lambda 인자: 식`은 한 줄 익명 함수이며, `map`+`lambda`+조건식으로 데이터 변환을 간결하게 쓴다.
