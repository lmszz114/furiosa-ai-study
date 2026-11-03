# 261102 복습 — NPU에 LLM 올려 쓰기(퓨리오사 서빙) · K-Fold 교차검증

> 어제 끝낸 NPU 서버 접속·콘다 환경 위에서, 오늘은 **NPU에 실제 LLM을 올려 API 서버로 띄우고, 파이썬에서 그 모델에 추론을 요청**해 본다. 파이썬 코드보다 세팅·테스트가 중심이다. 뒷부분은 머신러닝의 **K-Fold 교차검증**이다.

## 목차
1. 작업 환경 (SSH · 작업 폴더 · 가상환경)
2. NPU에 LLM 올리기 — 퓨리오사 LLM 서빙
3. test0001 — 환경 점검
4. f01_furiosa_test — NPU 서버에 추론 요청 (OpenAI 호환)
5. m02_kfold — K-Fold 교차검증
6. 한 줄 요약

---

## 1. 작업 환경 (SSH · 작업 폴더 · 가상환경)

- **접속**: VSCode의 SSH 기능으로 원격 NPU 서버(`furiosa-npu`)에 붙어, 서버 위에서 바로 코드를 편집·실행한다. (어제 만든 SSH 설정을 VSCode의 "Remote - SSH"가 그대로 쓴다.)
- **작업 폴더**: `root/furiosa-study` (원격 서버 안의 작업 디렉터리)
- **가상환경**: `furiosa02` (Python 3.11.18) — 어제 만든 콘다 환경. 터미널에서 `conda activate furiosa02`로 켠 뒤 작업한다.

즉 어제까지가 "접속 + 콘다 환경 준비"였다면, 오늘은 그 환경 안에서 NPU로 LLM을 돌리는 단계다.

---

## 2. NPU에 LLM 올리기 — 퓨리오사 LLM 서빙

퓨리오사 NPU(RNGD)에 LLM을 올려, 다른 프로그램이 API로 불러 쓸 수 있는 **서버**로 띄운다. 관련 문서는 퓨리오사 개발자 사이트에 있다(https://developer.furiosa.ai/ , 전제조건: /latest/en/get_started/prerequisites.html). `furiosa-llm`은 이미 설치돼 있다.

### 2-1. HuggingFace 토큰 발급 & 로그인

모델은 HuggingFace(모델 공유 플랫폼)에서 받아온다. 비공개·권한이 필요한 모델을 받으려면 **본인 토큰으로 로그인**해야 한다.

- 토큰 발급: huggingface.co 로그인 → Settings → Access Tokens → **Create New Access Token** → 생성된 토큰 복사
- CLI 로그인:
```bash
hf --version          # 2.4.0 설치 확인 (hf = HuggingFace CLI)
hf auth login         # 실행 후 'Paste an access token' 선택 → 복사한 토큰 붙여넣기 → 엔터
hf auth whoami        # 로그인된 사용자 확인
```

- **토큰은 HuggingFace에서 모델을 받을 권한을 증명하는 열쇠다.** 한 번 `hf auth login`으로 로그인해두면 이후 모델 다운로드 때 자동으로 인증된다. (⚠️ 토큰은 개인 비밀값이라 코드·문서에 그대로 적지 말 것.)

### 2-2. 모델을 NPU에 올려 서버 띄우기

```bash
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
```

- **`furiosa-llm serve <모델>`** 는 지정한 LLM을 **NPU(RNGD)에 적재하고, OpenAI 호환 API 서버를 띄우는** 명령이다. 실행하면 서버가 떠서 요청을 기다리는 상태가 된다(이 터미널은 서버가 점유하므로, 추론 코드는 다른 터미널에서 실행한다).
- **"OpenAI 호환"** 이 핵심이다. 지금까지 외부 OpenAI API를 쓰던 방식(OpenAI 클라이언트 코드)을 **주소만 이 로컬 서버로 바꿔 그대로** 쓸 수 있다는 뜻이다(4번에서 사용).
- **FP8**: 모델 이름 끝의 FP8은 가중치를 8비트로 줄인(양자화한) 경량 버전이라는 표시다. 메모리를 덜 쓰고 추론이 빠르다. 퓨리오사가 제공하는 모델은 여러 가지가 있다(아래 f01 주석 참고).

### 2-3. 상태 모니터링 · 캐시 확인

```bash
watch -n 1 furiosa-smi status   # NPU 상태를 1초마다 실시간 확인 (종료: Ctrl+C)
hf cache ls                     # HuggingFace로 받아 캐시에 저장된 모델 목록 확인
```

- 모델을 올리면 `furiosa-smi status`의 메모리 사용량·코어 사용률이 올라간다. 서빙·추론이 실제로 NPU를 쓰는지 여기서 확인한다.
- `hf cache ls`로 **이미 다운로드된 모델**을 확인할 수 있다(이미 받은 모델은 다시 받지 않는다).

---

## 3. test0001 — 환경 점검

```python
a = 1
b = 2
c = a + b
print(c)
```

내용은 단순 덧셈이다. 목적은 **SSH로 접속한 원격 서버에서, 올바른 가상환경(furiosa02)으로 파이썬이 제대로 실행되는지** 확인하는 것뿐이다. 새 환경에서 가장 먼저 "코드가 돌긴 하는가"를 점검하는 최소 테스트다.

---

## 4. f01_furiosa_test — NPU 서버에 추론 요청 (OpenAI 호환)

2번에서 띄운 로컬 NPU 서버에 파이썬으로 접속해 실제 추론을 받아 본다.

```python
"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8
"""   # ← 퓨리오사가 제공하는(serve 가능한) 모델 예시들

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",   # 외부가 아니라 '내 서버'(로컬 NPU 서버) 주소
    api_key="EMPTY"                         # 로컬 서버라 키가 필요 없음
)

models = client.models.list()     # 현재 서버에 올라와 있는 모델 목록
model_id = models.data[0].id      # 첫 번째(=지금 serve 중인) 모델의 id
print("모델 : ", model_id)

response = client.chat.completions.create(
    model = model_id,
    messages=[
        {"role" : "user", "content" : "NPU와 GPU의 차이점을 한국어로 설명해줘."}
    ],
    max_tokens=3000
)
print(response.choices[0].message.content)
```

- **지금까지 쓰던 OpenAI 클라이언트 코드와 똑같다. 다른 건 `base_url`뿐이다.** 외부 서비스 주소 대신 **내 NPU 서버(`http://127.0.0.1:8000/v1`)** 를 가리킨다. `127.0.0.1`은 "이 컴퓨터 자신"을 뜻하는 주소라, 같은 서버에서 띄운 LLM 서버로 요청이 간다.
- **`api_key="EMPTY"`**: 외부 API는 유료라 키가 필요했지만, 내가 띄운 로컬 서버는 인증이 없어 아무 값(EMPTY)이나 넣는다.
- **`client.models.list()`** 로 서버에 올라와 있는 모델을 확인하고, 그 id로 `chat.completions.create`를 호출해 답을 받는다. 호출 방식(messages의 role/content, max_tokens)은 외부 OpenAI API와 동일하다.
- 정리하면, **"내 NPU에 올린 LLM"을 마치 외부 API처럼 표준 방식으로 불러 쓰는** 실습이다. OpenAI 호환 서버라서 코드 재사용이 쉽다.

---

## 5. m02_kfold — K-Fold 교차검증

머신러닝 평가 방법인 **K-Fold 교차검증(Cross-Validation)** 을 다룬다.

### 왜 K-Fold인가

지금까지는 `train_test_split`으로 데이터를 train/test로 **한 번만** 나눠 평가했다. 그런데 "어쩌다 어떻게 나뉘었는지"에 따라 점수가 운 좋게/나쁘게 나올 수 있다. **K-Fold는 데이터를 K조각으로 나눠, 각 조각을 번갈아 테스트로 쓰고 나머지로 학습하기를 K번 반복**한다. K개의 점수를 평균 내므로, 한 번 나누기보다 **안정적이고 신뢰도 높은 평가**가 된다.

```
데이터를 5조각으로 나눔 (K=5)
1회차: [테스트][학습 ][학습 ][학습 ][학습 ]
2회차: [학습 ][테스트][학습 ][학습 ][학습 ]
3회차: [학습 ][학습 ][테스트][학습 ][학습 ]
4회차: [학습 ][학습 ][학습 ][테스트][학습 ]
5회차: [학습 ][학습 ][학습 ][학습 ][테스트]
→ 5번의 점수를 평균
```

### 5-1. m02_kfold_00_iris — 분류

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

x, y = load_iris(return_X_y=True)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)   # 5조각, 섞어서 나눔

model = DecisionTreeClassifier()
scores = cross_val_score(model, x, y, cv=kfold)   # K번 학습·평가를 한 번에

print('acc :', scores, '\n평균 acc :', round(np.mean(scores), 2))
```

- **`KFold(n_splits=5, shuffle=True, random_state=333)`**: 데이터를 5조각으로 나누는 방법을 정의한다. `shuffle=True`는 나누기 전에 섞어 순서 편향을 없애고, `random_state`로 매번 같은 방식으로 섞이게 고정한다.
- **`cross_val_score(model, x, y, cv=kfold)`**: 정의한 K-Fold 방식으로 **K번 학습·평가를 자동으로** 수행하고 점수 배열을 돌려준다. 그래서 이 방식에서는 `train_test_split`도, 직접 `fit`도 쓰지 않는다 — 전체 `x, y`를 넘기면 내부에서 알아서 나눠 돌린다.
- 결과는 5개의 점수(각 회차 정확도)와 그 평균이다. 분류 모델이라 점수는 정확도(accuracy)다.

### 5-2. m02_kfold_01_california — 회귀

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

x, y = fetch_california_housing(return_X_y=True)   # 캘리포니아 집값 (회귀)

kfold = KFold(n_splits=5, shuffle=True, random_state=333)
model = DecisionTreeRegressor()
scores = cross_val_score(model, x, y, cv=kfold)

print('loss :', scores, '\n평균 loss :', round(np.mean(scores), 2))
```

- 회귀 데이터(캘리포니아 집값)에 **회귀 모델(`DecisionTreeRegressor`)** 을 쓴 것 외에 K-Fold 적용 방식은 동일하다. 분류든 회귀든 `cross_val_score`로 똑같이 교차검증한다.
- **보충(정확한 이해를 위해)**: 코드에서 출력 라벨을 `loss`라고 적었지만, `cross_val_score`가 회귀 모델에 매기는 기본 점수는 **R²**다(손실값이 아니다). 즉 출력되는 숫자는 1에 가까울수록 좋은 R²이며, 라벨만 `loss`로 표기된 것이다. (다른 지표로 보려면 `cross_val_score(..., scoring=...)`로 지정한다.)

---

## 6. 한 줄 요약

> **NPU LLM 서빙**: 어제 만든 환경 위에서, HuggingFace 토큰으로 로그인(`hf auth login`)한 뒤 `furiosa-llm serve <모델>`로 NPU에 LLM을 올려 **OpenAI 호환 API 서버**를 띄운다. 그러면 기존 OpenAI 클라이언트 코드를 **`base_url`만 로컬 서버(`http://127.0.0.1:8000/v1`, `api_key="EMPTY"`)로 바꿔** 그대로 추론에 쓸 수 있다(`client.models.list()`로 모델 확인 → `chat.completions.create`). **K-Fold 교차검증**은 데이터를 K조각으로 나눠 번갈아 테스트하며 K번 평가해 평균 내는 방법으로, 한 번 나누는 `train_test_split`보다 안정적이다. `KFold(n_splits, shuffle, random_state)`로 나누는 방식을 정하고 `cross_val_score(model, x, y, cv=kfold)`로 K번 학습·평가를 한 번에 수행한다(train_test_split·fit을 따로 쓰지 않음). 분류는 정확도, 회귀는 R²로 평가된다.
