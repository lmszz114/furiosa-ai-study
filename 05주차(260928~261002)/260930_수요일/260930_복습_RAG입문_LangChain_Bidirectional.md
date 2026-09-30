# 2026.09.30 복습 — RAG 입문(LangChain·API 키 관리·LCEL) & 양방향 RNN(Bidirectional)

> 오늘 흐름: RAG 이론 → LangChain 가상환경 구축 → API 키를 다루는 4가지 방법(하드코딩→.env) → 프롬프트·체인(LCEL)·출력파서 → (keras) 시계열 타깃을 T로 변경 → 양방향 RNN(Bidirectional).

---

## 1. RAG란 (이론 배경)

- **RAG (Retrieval-Augmented Generation, 검색 증강 생성)**: LLM이 답을 만들 때, 외부 문서·데이터를 **검색(retrieval)** 해 가져와 그 내용을 **근거로 답을 생성(generation)** 하는 방식이다.
- 목적: LLM이 모르는 최신 정보·내부 문서를 보완하고, 근거 없는 지어내기(환각)를 줄인다.
- **오늘 실습의 위치**: 오늘 코드는 RAG의 **기반 도구인 LangChain으로 LLM을 호출하고, 프롬프트·체인·출력파서를 다루는 단계**다. (문서를 검색해 붙이는 retrieval 단계는 아직 아니며, 그 앞의 뼈대를 익히는 과정이다.)

---

## 2. 가상환경 생성 및 설치 (LangChain)

RAG 실습용으로 **새 가상환경**을 만든다. (기존 keras 환경과 분리 — 라이브러리 충돌 방지.)

```bash
# cmd에서

# ① langchain 가상환경 생성 (python 3.11)
conda create -n langchain python=3.11

# ② 활성화
activate langchain          # (conda activate langchain 도 가능)

# ③ 라이브러리 설치
pip install langchain
pip install langchain_openai
pip install python-dotenv    # .env 파일 읽기용 (dotenv 코드에서 사용)
```

- **주의(사실 정정)**: 메모에는 `pip create -n ...`으로 적혀 있으나, **가상환경 생성은 `conda create`** 명령이다. `pip`에는 `create` 기능이 없다(pip은 패키지 설치 담당). miniconda를 쓰므로 `conda create`가 맞다.
- `python-dotenv`는 메모엔 없지만 뒤 코드(`from dotenv import load_dotenv`)에서 쓰이므로 설치해야 한다.

### VSCode에서 가상환경 잡기
새 환경이 VSCode 인터프리터 목록에 안 뜰 때:
1. VSCode 껐다 켜기, 또는
2. `Ctrl+Shift+P` → 검색창에 `reload` 입력 → **Developer: Reload Window** 선택.

---

## 3. API 키 다루는 4가지 방법 (rag01 → rag05)

OpenAI 같은 LLM API는 **키(비밀번호 같은 것)** 가 필요하다. 이 키를 **어디에 두느냐**를 나쁜 방법 → 좋은 방법 순으로 배운다. 핵심 원칙: **키는 코드에 노출되면 안 된다.**

### rag01 — 코드에 키 직접 넣기 (❌ 가장 나쁨)
```python
from langchain_openai import ChatOpenAI

openai_api_key = 'sk-proj-****(마스킹)****'   # 코드에 그대로 노출

llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0,
                 openai_api_key=openai_api_key)

response = llm.invoke('내가 누구게?')   # 질문 한 번 보내기
print(response.content)               # 답 텍스트 출력
```
- **`ChatOpenAI(...)`**: LLM을 불러오는 객체. `temperature=0` = 출력 무작위성 0(가장 일관·확정적인 답).
- **`llm.invoke("...")`**: 질문을 한 번 보내고 답을 받는다. (keras의 `predict`에 해당.) **invoke는 1회성** — 이전 대화를 기억하지 않는다(메모리에 저장 안 됨).
- **`response.content`**: 답의 본문 텍스트.
- 문제: 키가 코드에 그대로 있어 깃허브 등에 올라가면 **유출**된다. 실무에선 절대 이렇게 안 한다.

### rag02 — os.environ으로 코드에서 환경변수 설정 (❌ 여전히 노출)
```python
import os
os.environ["OPENAI_API_KEY"] = 'sk-proj-****'   # 코드에서 환경변수로 등록
llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0)  # 키 인자 생략 가능
```
- `OPENAI_API_KEY`라는 **약속된 이름**의 환경변수에 키를 넣으면, `ChatOpenAI`가 자동으로 그 값을 찾아 쓴다(그래서 `openai_api_key=` 인자 생략).
- 그러나 **키가 여전히 코드 안에 있어** 노출 문제는 그대로다.

### rag03 — 시스템 환경변수에 직접 등록 (코드에서 제거)
```python
# os.environ[...] 줄을 지우고, 키는 윈도우 '시스템 환경 변수'에 직접 등록
llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0)
```
- 키를 OS(윈도우) 환경변수에 등록해두면 코드에 키 문자열이 사라진다. 코드가 깔끔해지지만, 팀·다른 PC로 옮길 때 매번 OS에 등록해야 하는 불편이 있다.

### rag04 — 키가 잘 등록됐는지 확인
```python
import os
key = os.getenv("OPENAI_API_KEY")

if key is None:
    print("OPENAI_API_KEY 없음")
else:
    print("키 길이: ", len(key))
    print("키 확인: ", key[:8] + "..." + key[-4:])   # 앞 8·뒤 4자리만 (전체 노출 방지)
```
- **`os.getenv("이름")`**: 환경변수 값을 읽어온다. 없으면 `None`.
- 키 전체를 print하면 그것도 노출이라, **앞뒤 일부만** 잘라 확인한다.

### rag05 — .env 파일 + python-dotenv (✅ 실무 방식)
```python
from dotenv import load_dotenv
load_dotenv()   # 같은 폴더의 .env 파일을 읽어 환경변수로 로드

llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0)
```
- **`.env` 파일**에 키를 적어두고, **`load_dotenv()`** 로 프로그램 실행 시 불러온다. 키가 **코드와 분리**되고, `.env`는 깃에 안 올리면 되니 안전하다.
- 참고: VSCode에선 `load_dotenv()` 없이도 되는 경우가 있으나, **다른 환경에선 안 되므로 기본적으로 써주는 게 맞다.**

### .env 파일과 .gitignore
```
# .env  (키를 여기에 보관)
OPENAI_API_KEY = sk-proj-****
MONOROUTER_API_KEY = ****
```
```
# .gitignore  (이 파일들을 깃 추적에서 제외)
.env
```
- **`.gitignore`에 `.env`를 넣어** 키 파일이 깃허브에 안 올라가게 막는다. RAG 실습에서 키 유출을 막는 핵심 습관이다.

> ⚠️ **주의**: `.gitignore`는 `.env`만 막는다. **rag01·rag02처럼 키를 코드(.py)에 직접 박아두면 그 파일은 그대로 깃허브에 올라간다.** 실습 후 그 파일들의 키는 지우거나 `.env` 방식으로 바꾸는 것이 안전하다. (특히 Public 레포일 때.)

---

## 4. rag06 — 서드파티 API 사용 (monogpt)

OpenAI 공식이 아니라 다른 제공처(monogpt) API를 쓸 때는 **키 + 접속 주소(base_url)** 둘 다 필요하다.
```python
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()   # .env에서 키 읽기
base_url = "https://monogpt.kr/api/monorouter/v1"     # 접속 주소

llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0,
                 api_key=api_key, base_url=base_url)
```
- **`base_url`**: 요청을 보낼 서버 주소. 공식 OpenAI가 아니면 이 주소를 바꿔줘야 그쪽으로 연결된다.
- **`.strip()`**: 키 앞뒤에 실수로 들어간 공백·줄바꿈을 제거(인증 오류 방지).

---

## 5. rag07 — PromptTemplate (프롬프트 템플릿)

프롬프트에 **나중에 채울 빈칸(변수)** 을 두는 틀이다.
```python
from langchain_core.prompts import PromptTemplate

template = "{country}의 수도는 어디인가요?"
prompt_template = PromptTemplate.from_template(template)

print(prompt_template)
# input_variables=['country'] ... template='{country}의 수도는 어디인가요?'
```
- **`{country}`**: 실행할 때 값을 채워 넣을 변수 자리(중괄호).
- **`PromptTemplate.from_template(문자열)`**: 그 문자열을 템플릿으로 만든다. `input_variables`에 `{}` 변수가 자동 인식된다.
- 약어 정리(메모): **pmo = prompt · model · output**, **plp = prompt · llm · parser**. 프롬프트 → 모델 → 출력(파서)로 이어지는 흐름을 가리킨다.

---

## 6. rag08 — LCEL 체인 (prompt | model)

**LCEL(LangChain Expression Language)**: 여러 단계를 **파이프 `|`** 로 연결해 하나의 체인(chain)으로 만드는 문법이다.
```python
prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해줘.")

chain = prompt | model          # 프롬프트 → 모델 연결

input = {"topic": "양자컴퓨터 학습 원리"}
response = chain.invoke(input)  # 실행 (model.predict 같은 것)
print(response.content)
```
- **`chain = prompt | model`**: 프롬프트에 값을 채워 모델에 넘기는 과정을 `|`로 연결. 데이터가 왼쪽 → 오른쪽으로 흐른다.
- **`chain.invoke(input)`**: 딕셔너리로 변수 값을 넣어 실행. `{"topic": "..."}`가 `{topic}` 자리에 들어간다.

### rag08_02 — 변수 2개
```python
prompt = PromptTemplate.from_template("{topic}에 대해 {how} 설명해줘.")
input = {"topic": "양자컴퓨터 학습 원리", "how": "초등학생도 이해하기 쉽게"}
```
- 변수를 여러 개 두고 딕셔너리로 각각 채운다. 프롬프트를 유연하게 조립할 수 있다.

---

## 7. rag09 — OutputParser (출력 파서)

모델 응답에서 **`.content`를 매번 꺼내지 않고, 바로 문자열로 받게** 해주는 단계다.
```python
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해줘.")
chain = prompt | model | output_parser   # 파서까지 연결

response = chain.invoke({"topic": "양자컴퓨터 학습 원리"})
print(response)   # response.content 아니라 바로 response (문자열)
```
- **`StrOutputParser`**: 모델 출력을 **깔끔한 문자열**로 바꿔준다. 체인 끝에 `| output_parser`를 붙이면 `response`가 바로 텍스트라 `.content`가 필요 없다.
- 이것이 **prompt → model → output_parser** 3단 체인의 완성형이다.

### rag09_02 — 역할 부여 + 출력 형식 지정
```python
template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화를 작성해주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

# 상황
{question}

# FORMAT
- 영어회화:
- 한글번역:
"""
...
input = {"question": "강남역에서 판교역까지 가장 빠르게 갈 수 있는 방법이 궁금해요."}
```
- 프롬프트에 **역할(영어 선생님)** 과 **출력 형식(FORMAT)** 을 지정하면, 모델이 그 틀에 맞춰 답한다. 프롬프트 설계(프롬프트 엔지니어링)의 기본이다.

---

## 8. (keras) keras58 — 예측 타깃을 wd → T로 변경

어제 jena 실습에서 y를 **wd(풍향) → T(온도)** 로 바꿔 다시 돌린다.
```python
# T (degC)는 index_col=0으로 Date Time이 빠진 뒤 기준 1번 컬럼
x = np.delete(bbb[:-144], 1, axis=2)   # T(1번) 컬럼 제외 → (…, 144, 13)
y = bbb[:-144, -1, 1]                   # 마지막 timestep의 T

# 예측도 동일하게
x_predict = np.delete(bbb[-144:], 1, axis=2)
y_true    = bbb[-144:, -1, 1]
```
- **타깃을 바꾸는 것 = "어느 컬럼을 x에서 빼고(정보 누수 방지), 어느 컬럼을 y로 쓸지"만 바꾸는 것.**
- T는 중간(1번) 컬럼이라 슬라이싱(`:-1`)으로 못 빼고 **`np.delete(..., 1, axis=2)`** 로 제거한다. (wd는 맨 끝이라 `:-1`로 뺐던 것과 차이.)
- 컬럼 수는 14→13로 동일해서 **모델·스케일링·input_shape는 그대로**.
- **왜 index가 1인가**: `read_csv(..., index_col=0)`이 "Date Time"을 인덱스로 빼서, 남은 컬럼이 p=0, **T=1**, …, wd=13이 된다. (원본 CSV로 세면 T가 2번이지만 Date Time이 인덱스로 빠져 한 칸 당겨짐.)

---

## 9. (keras) keras59 — 양방향 RNN (Bidirectional)

### Bidirectional이란
- **Bidirectional(양방향)**: 시퀀스를 **앞→뒤 방향뿐 아니라 뒤→앞 방향으로도** 읽어, 양쪽 맥락을 함께 본다.
- **모델이 아니라 "래퍼(wrapper)"** 다. 기존 RNN 층(LSTM 등)을 **감싸는** 형태로 쓴다.
```python
from tensorflow.keras.layers import Bidirectional, LSTM

model.add(Bidirectional(LSTM(64), input_shape=(3,1)))   # LSTM을 양방향으로 감쌈 (괄호 주의)
```
- **괄호 주의**: `Bidirectional(LSTM(64), input_shape=(3,1))` — `LSTM(64)`을 Bidirectional의 인자로 넣고, `input_shape`는 Bidirectional 쪽에 준다.
- **파라미터가 2배**가 된다(정방향 + 역방향 각각 학습하므로). `model.summary()`로 확인한다.

### 성능 (실행 결과)
- **keras59_Bidirectional2** ([50,60,70]→80 예측): 일반 LSTM 79.85 → **양방향 73.12**. 즉 **오히려 더 나빠졌다.**
- **keras59_Bidirectional3_jena** (T 예측, 양방향): **R² 약 0.9987, RMSE 약 0.117**, 약 3347초 소요.
- **해석**:
  - Bidirectional을 쓴다고 **항상 좋아지지 않는다**(위 2번 사례에서 하락). return_sequences·층 쌓기와 마찬가지로, 데이터·문제에 따라 효과가 다르다.
  - jena에서 R²가 어제(wd, -0.13)보다 극적으로 좋아진(0.9987) 것은 **주로 타깃을 T로 바꾼 효과**로 보인다(*분석/추측*). T(온도)는 시간에 따라 완만·연속적으로 변해 예측이 쉬운 반면, wd(풍향)는 각도(0~360)라 359°≈1°의 불연속 때문에 회귀가 불리했다. (양방향이 얼마나 기여했는지는 이 실습만으로 분리 확인되지 않음.)

---

## 10. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| rag01_insert_key | 키 하드코딩 | 코드에 키 직접(노출, 나쁨). `invoke`는 1회성 |
| rag02_environ01 | os.environ | 코드에서 환경변수 등록(여전히 노출) |
| rag03_env_key | 시스템 환경변수 | 키를 OS에 등록, 코드에서 제거 |
| rag04_env_key_check | 키 확인 | `os.getenv`, 앞뒤 일부만 출력 |
| rag05_env | .env + dotenv | `load_dotenv()`로 .env 로드(실무 방식) |
| .env / .gitignore | 키 보관·제외 | 키는 .env에, .gitignore로 깃 제외 |
| rag06_monogpt | 서드파티 API | key + base_url 둘 다 필요 |
| rag07_prompt | PromptTemplate | `{변수}` 빈칸 템플릿, from_template |
| rag08_LCEL01/02 | LCEL 체인 | `prompt \| model`, invoke, 변수 여러 개 |
| rag09_output_parser01/02 | 출력 파서 | `\| StrOutputParser`로 바로 문자열, 역할·FORMAT 지정 |
| keras58 (T 변경) | 타깃 교체 | np.delete로 T 제외, y=…,1 |
| keras59_Bidirectional1/2 | 양방향 RNN | LSTM을 Bidirectional로 감쌈(래퍼), 파라미터 2배, 성능은 케바케 |
| keras59_Bidirectional3_jena | 양방향+실데이터 | T 예측 R² 약 0.9987 |

---

## 11. 오늘 한 줄 요약

> **RAG**는 외부 문서를 검색해 근거로 답하는 방식이며, 오늘은 그 기반인 **LangChain**을 다뤘다. LLM은 `ChatOpenAI` + `invoke`로 호출하고, **API 키는 코드에 노출하지 않는 것이 핵심** — 하드코딩(rag01) → `os.environ`(rag02) → 시스템 환경변수(rag03) → **`.env` + `load_dotenv()`(rag05, 실무)** 순으로 배웠고, `.gitignore`로 `.env`를 깃에서 제외한다.
> **LCEL**은 `prompt | model | output_parser`처럼 `|`로 단계를 잇는 문법이고, `PromptTemplate`의 `{변수}`를 `invoke({...})`로 채운다. `StrOutputParser`를 붙이면 응답이 바로 문자열로 나온다.
> (keras) 예측 타깃을 **wd→T로 교체**(np.delete로 컬럼 제외), **Bidirectional**은 시퀀스를 양방향으로 읽는 **래퍼**(모델 아님, 파라미터 2배)다. 다만 양방향이 항상 성능을 올리진 않으며(예: 79.85→73.12), jena의 큰 개선은 주로 타깃을 T로 바꾼 영향으로 보인다(*추측*).
