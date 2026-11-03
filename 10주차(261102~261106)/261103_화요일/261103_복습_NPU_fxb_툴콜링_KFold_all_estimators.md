# 261103 복습 — NPU 모델 컴파일(fxb)·툴콜링 · K-Fold 올바른 사용 · 전체 모델 자동 비교

> 앞부분은 NPU에서 모델을 **직접 컴파일(fxb)** 해 서빙하고, 그 모델로 **툴 콜링(함수 호출)** 을 해 본다. 뒷부분은 머신러닝에서 **K-Fold를 올바른 자리에 쓰는 법**(전체 vs train만)과, **사이킷런의 모든 모델을 한 번에 돌려 비교**하는 방법이다.

> ⚠️ 명령어 실행 위치: 아래 셸 명령어들은 **PowerShell(내 PC)이 아니라, SSH로 접속한 원격 NPU 서버(리눅스 bash)에서** 실행한다. PowerShell/electerm/VSCode는 그 서버에 접속하는 통로일 뿐이고, 명령 자체는 서버 안에서 돈다.

## 목차
1. NPU 준비 — 셸 설정 & 빌드 도구 설치
2. fxb — 모델을 NPU용으로 컴파일하기
3. NPU에 모델 서빙하기
4. f03_tool_calling — NPU 모델로 툴 콜링(함수 호출)
5. f04_fxb — 컴파일된 모델 직접 불러오기
6. K-Fold 올바르게 쓰기 — 전체(m02) vs train만(m03)
7. all_estimators — 모든 모델 한 번에 비교 (m04·m05)
8. 한 줄 요약

---

## 1. NPU 준비 — 셸 설정 & 빌드 도구 설치

### 1-1. 로그인 셸에서도 설정이 적용되게

```bash
grep -q 'source ~/.bashrc' ~/.bash_profile 2>/dev/null || echo 'source ~/.bashrc' >> ~/.bash_profile
```

- **하는 일**: `.bash_profile`에 `source ~/.bashrc` 줄이 있는지 조용히 검사하고, **없으면 추가**한다.
- **한 줄 뜯어보기**:
  - `grep -q '...' ~/.bash_profile` : 그 문구가 파일에 있는지 검색(`-q`는 결과를 화면에 안 띄우고 "있다/없다"만 판단).
  - `2>/dev/null` : 파일이 없을 때 뜨는 에러 메시지를 버린다(화면을 깔끔하게).
  - `|| echo '...' >> ~/.bash_profile` : 앞 검색이 **실패(=문구 없음)** 하면(`||`), 그 문구를 파일 끝에 **추가**(`>>`)한다.
- **왜**: 로그인 방식(로그인 셸)으로 접속하면 `.bashrc`가 자동으로 안 읽히는 경우가 있다. 이러면 conda 설정 등이 적용 안 돼 `conda` 명령이 또 안 먹는다. `.bash_profile`이 `.bashrc`를 불러오게 만들어, **어떻게 접속하든 설정이 항상 적용**되게 하는 것이다. (어제 겪은 "conda가 안 잡힘" 문제의 근본 예방책)

### 1-2. 빌드 도구(크로스 컴파일러) 설치

```bash
apt-get update
apt-get install --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross
```

- **`apt-get update`**: 설치 가능한 패키지 목록을 최신으로 갱신한다(실제 설치는 아직 안 함).
- **`apt-get install ...`**: 모델을 NPU용으로 빌드할 때 필요한 **크로스 컴파일러**를 설치한다.
  - `gcc-aarch64-linux-gnu`, `libc6-dev-arm64-cross` : **aarch64(ARM64) 아키텍처용** 컴파일러와 라이브러리. NPU 빌드 과정이 이 아키텍처용 코드를 만들어야 해서 필요하다.
  - `--no-install-recommends` : 꼭 필요한 핵심 패키지만 깔고, 곁다리로 딸려오는 권장 패키지는 설치하지 않는다(가볍게).
- (크로스 컴파일 = 지금 돌고 있는 환경이 아니라 **다른 아키텍처(여기선 ARM64)에서 돌 코드를 만드는 것**.)

---

## 2. fxb — 모델을 NPU용으로 컴파일하기

**fxb는 FuriosaAI의 모델 빌드(컴파일) 도구**다. GPU와 달리 NPU(RNGD)는 모델을 **NPU 하드웨어에 맞게 컴파일**해 둬야 효율적으로 돌아간다. HuggingFace 모델을 받아 NPU용 결과물(`.fxb` 파일)로 변환하는 과정이다.

```bash
# 모델을 NPU용으로 컴파일 → qwen3-8b-fp8.fxb 생성
fxb build Qwen/Qwen3-8B-FP8 qwen3-8b-fp8

# 실제로 빌드하지 않고, 무엇을 할지 미리 점검만
fxb build Qwen/Qwen3-8B-FP8 qwen3-8b-fp8 --dry-run

# 빌드한 .fxb를 등록(캐시에 추가)
fxb add ./qwen3-8b-fp8.fxb

# 컴파일 캐시 목록 확인
fxb cache ls
```

- **`fxb build <HF모델> <출력이름>`**: HuggingFace 모델(`Qwen/Qwen3-8B-FP8`)을 NPU용으로 컴파일해 `<출력이름>.fxb`를 만든다.
- **`--dry-run`**: 실제 빌드(오래 걸림)를 하기 전에 **어떤 작업이 일어날지 미리 확인**하는 "모의 실행" 옵션이다. 설정이 맞는지 점검용.
- **`fxb add ./*.fxb`**: 만든 fxb를 등록해 이후 서빙·로드에서 쓸 수 있게 한다.
- **`fxb cache ls`**: 이미 컴파일해 둔 fxb 목록을 본다(한 번 컴파일하면 재사용).
- **왜 중요**: 컴파일은 오래 걸리므로 **한 번 fxb로 빌드해 두면, 다음부터는 그 fxb로 바로 서빙**해 시간을 아낀다(아래 3-2).

---

## 3. NPU에 모델 서빙하기

### 3-1. HuggingFace 모델을 바로 서빙 (옵션 추가)

```bash
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 0.0.0.0 --port 8000
```

어제 배운 `furiosa-llm serve`에 옵션이 붙었다.
- **`--devices npu:5`**: 여러 NPU 중 **5번 장치**에 모델을 올린다(장치를 지정해 분산·충돌 방지).
- **`--host 0.0.0.0`**: 이 서버의 **모든 네트워크 주소에서 접속을 허용**한다. `127.0.0.1`(자기 자신만)과 달리 **외부(다른 기기)에서도** 이 API에 접속할 수 있게 연다.
- **`--port 8000`**: API 서버가 쓸 포트 번호.

### 3-2. 컴파일한 로컬 fxb로 서빙

```bash
furiosa-llm serve Qwen/Qwen3-8B-FP8 --fxb ./qwen3-8b-fp8.fxb
```

- **`--fxb ./*.fxb`**: 매번 새로 컴파일하지 않고, **2번에서 미리 빌드해 둔 fxb를 불러와** 바로 서빙한다. 컴파일 단계를 건너뛰어 **빠르게 띄울 수 있다.**

---

## 4. f03_tool_calling — NPU 모델로 툴 콜링(함수 호출)

NPU에 올린 LLM에게 **"필요하면 이 함수들을 써라"** 고 알려주고, LLM이 **어떤 함수를 어떤 인자로 호출할지** 정하게 하는 실습이다. (LangGraph에서 배운 도구 호출의 "날것" 버전 — OpenAI API를 직접 쓴다.)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # 내 NPU 서버

# 실제 파이썬 함수들
def get_weather(location:str, unit:str):
    return f"Getting the weather for {location} in {unit}"
def get_time(timezone:str):
    return f"Getting the current time in {timezone}..."

# LLM에게 알려줄 '함수 설명서'(스펙)
tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": { "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state..."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location", "unit"] }}},
    { ... get_time ... }
]

response = client.chat.completions.create(
    model = client.models.list().data[0].id,
    messages = [{"role":"user", "content":"오늘 판교의 날씨는 어때?"}],
    tools = tools,                                                   # 쓸 수 있는 함수 목록
    tool_choice = {"type":"function", "function":{"name":"get_weather"}},  # 이 함수를 쓰도록 지정
    temperature=0.0,
)

tool_call = response.choices[0].message.tool_calls[0].function
print(tool_call.name)        # 호출된 함수 이름: get_weather
print(tool_call.arguments)   # LLM이 뽑아낸 인자(JSON): {"location":"판교...","unit":"celsius"}
print(get_weather(**json.loads(tool_call.arguments)))   # 그 인자로 실제 함수 실행
```

- **`tools`**: LLM에게 주는 "함수 설명서"다. 함수 이름·설명·인자(타입·필수 여부)를 JSON으로 적는다. LLM은 이 설명을 보고 "이 질문엔 어떤 함수가 맞는지, 인자는 뭘 넣을지"를 판단한다.
- **`tool_choice`**: 어떤 함수를 쓸지 **강제 지정**한다(여기선 `get_weather` 고정). 지정하지 않으면 LLM이 알아서 고른다.
- **핵심 흐름**: LLM은 함수를 **직접 실행하지 않는다.** "`get_weather`를 `location=판교, unit=celsius`로 불러라"는 **지시(이름+인자)만** 만들어 준다(`tool_calls`). 그 인자를 꺼내(`json.loads`) **실제 파이썬 함수는 내가 실행**한다. "LLM은 무엇을 호출할지 정하고, 실행은 코드가 한다"가 툴 콜링의 핵심이다.
- `temperature=0.0`은 매번 같은(일관된) 판단을 하도록 창의성을 0으로 둔 것이다.

---

## 5. f04_fxb — 컴파일된 모델 직접 불러오기

```python
from furiosa_llm import LLM
llm = LLM("Qwen/Qwen3-8B-FP8", fxb="/root/qwen3-8b-fp8.fxb")
```

- API 서버(`serve`)를 거치지 않고, 파이썬 코드에서 **`furiosa_llm`의 `LLM` 클래스로 컴파일된 fxb를 직접 불러와** 쓰는 방식이다. `fxb=` 에 2번에서 만든 파일 경로를 준다. (서버를 띄우는 대신 코드 안에서 모델을 바로 다루고 싶을 때.)

---

## 6. K-Fold 올바르게 쓰기 — 전체(m02) vs train만(m03)

어제 K-Fold를 배웠는데, **어떤 데이터에 적용하느냐**가 중요하다. 오늘은 캐글 실데이터(자전거 수요=회귀, Santander=분류)로 두 방식을 비교한다.

### 공통 — 캐글 데이터 전처리

```python
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # 0번 컬럼을 인덱스로
print(train_csv.isna().sum())      # 결측치(빈 값) 개수 확인 (isna = isnull)
x = train_csv.drop(['casual','registered','count'], axis=1)   # 입력에서 제외할 컬럼 빼기
y = train_csv['count']                                        # 타깃 컬럼

scaler = RobustScaler()            # 스케일러 4종 중 선택 (이상치에 강한 Robust)
scaler.fit(x_train); x_train = scaler.transform(x_train); x_test = scaler.transform(x_test)
```

- **`pd.read_csv(..., index_col=0)`**: CSV를 불러오며 첫 컬럼을 인덱스로. **`isna().sum()`/`isnull().sum()`** 으로 결측치 개수를 확인한다(둘은 같은 기능).
- **`drop([...], axis=1)`**: 입력 `x`에서 타깃이나 쓰면 안 되는 컬럼을 뺀다(자전거: `casual·registered`는 정답 누수라 제외, `count`가 타깃).
- **스케일러 4종**(`MinMax·Standard·MaxAbs·Robust`)을 바꿔 가며 쓸 수 있고, 여기선 이상치에 강한 `RobustScaler`를 골랐다.

### 6-1. m02 — 전체 데이터로 교차검증

```python
scores = cross_val_score(model, x, y, cv=kfold)   # ★ 전체 x, y 를 넣음
```

- 전체 데이터를 K-Fold로 교차검증한다. 모델의 대략적 성능을 빠르게 가늠하는 용도다. (단, 최종 평가용으로 떼어둔 데이터가 없다.)

### 6-2. m03 — train/test를 나눈 뒤 train만 교차검증 (+ test로 최종 평가)

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y, ...)
# ...스케일링...

scores = cross_val_score(model, x_train, y_train, cv=kfold)   # ★ train 으로만 교차검증
print('평균 acc :', round(np.mean(scores),2))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold) # test 예측값 모으기
acc = accuracy_score(y_test, y_predict)                        # 실제값 vs 예측값
```

- **차이의 핵심**: m02는 **전체**로, m03은 **train만** 교차검증하고 **test는 따로 남겨 최종 평가**에 쓴다. m03이 더 올바른 방식이다 — 교차검증은 train 안에서 하고, 모델이 한 번도 못 본 test로 마지막에 확인해야 **진짜 일반화 성능**을 알 수 있다. (어제 말한 "검증셋에 과적합" 문제를 피하는 구조)
- **`cross_val_predict`**: 교차검증을 돌리면서 **각 샘플의 예측값을 모아서** 돌려준다. `cross_val_score`가 "점수"만 준다면, 이건 "예측값"을 줘서 `accuracy_score`/`r2_score`로 실제값과 직접 비교할 수 있다.
- 분류(iris·Santander)는 `accuracy_score`, 회귀(자전거)는 `r2_score`로 평가한다.

---

## 7. all_estimators — 모든 모델 한 번에 비교 (m04·m05)

사이킷런에 있는 **모든 모델을 한꺼번에 가져와 전부 돌려 보고** 어떤 게 이 데이터에 잘 맞는지 비교하는 방법이다.

```python
from sklearn.utils import all_estimators
import warnings; warnings.filterwarnings('ignore')   # 수많은 경고 숨기기

all = all_estimators(type_filter='classifier')   # 모든 분류 모델 (regressor면 회귀)
print('모델의 개수 :', len(all))   # 분류 44개 / 회귀 55개 (sklearn 1.9.1 기준)

for name, model in all:
    try:
        model = model()
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, '모델 오류')   # 안 돌아가는 모델은 건너뜀
```

- **`all_estimators(type_filter='classifier' / 'regressor')`**: 사이킷런의 모든 분류/회귀 모델을 `(이름, 모델클래스)` 튜플 리스트로 가져온다(분류 44·회귀 55개).
- **for문으로 전부 학습·평가**해 성능을 한눈에 비교한다. 모델마다 요구 조건이 달라 일부는 에러가 나므로, **`try/except`로 실패한 모델은 건너뛰고** 계속 진행한다. `warnings.filterwarnings('ignore')`로 쏟아지는 경고를 숨긴다.
- **m04 vs m05의 차이**:
  - **m04**: `train_test_split` 후 `fit`/`score`로 **한 번** 평가(단순·빠름).
  - **m05**: 각 모델을 **K-Fold 교차검증**(`cross_val_score`)으로 평가 → 더 안정적인 비교.
- **목적**: "어떤 모델을 쓸지" 감이 없을 때, 전부 돌려 상위 후보를 추린 뒤 그것만 집중 튜닝하는 **모델 탐색(1차 스크리닝)** 용이다.
- (보충: m04의 `regressor` 버전은 iris(분류 데이터)에 회귀 모델을 돌리는 형태라 실제 문제로는 맞지 않지만, `all_estimators` 사용법을 익히는 실습으로 보면 된다.)

---

## 8. 한 줄 요약

> **NPU**: `fxb build`로 HF 모델을 NPU용으로 **컴파일(.fxb)** 해 두면 `furiosa-llm serve --fxb`로 빠르게 서빙할 수 있다(서빙 옵션 `--devices npu:5`로 장치 지정, `--host 0.0.0.0`로 외부 접속 허용). 셸 명령은 전부 **원격 서버에서** 실행하며, `.bash_profile`에 `source ~/.bashrc`를 넣어 접속 방식과 무관하게 설정이 적용되게 한다. NPU 모델로 **툴 콜링**도 되는데, LLM은 **어떤 함수를 어떤 인자로 호출할지만 정하고(tools/tool_choice→tool_calls), 실제 실행은 코드가** 한다. **K-Fold**는 전체(m02)보다 **train/test를 나눠 train만 교차검증하고 test로 최종 평가**(m03)하는 것이 올바르며, `cross_val_predict`로 예측값을 모아 실제값과 비교한다. **all_estimators**는 사이킷런의 모든 분류/회귀 모델(44·55개)을 for문으로 한 번에 돌려(try/except로 오류 모델은 건너뜀) 어떤 모델이 잘 맞는지 1차로 비교하는 도구다.
