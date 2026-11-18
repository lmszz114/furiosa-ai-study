# 261118 복습 — LangChain Tool Calling · 파이썬 데코레이터

> 앞부분은 LLM이 직접 못 하는 일(정확한 계산 등)을 **도구(함수)에 맡기는 Tool Calling**이다. `@tool`로 함수를 도구로 만들고, LLM이 "어떤 도구를 어떤 값으로 부를지" 고르면 코드가 실제로 실행한다. 뒷부분은 그 `@tool`의 정체인 **파이썬 데코레이터** 문법이다.

## 목차
**A. Tool Calling**
1. f31 — `@tool` 정의 · `bind_tools` · `tool_calls`
2. f32 — tool_calls 뜯어보기(name·args·id)
3. f33 — 도구 실제 실행(`tool.invoke`)
4. f34 — 여러 도구와 tool_map
**B. 파이썬 데코레이터**
5. p03 — 데코레이터 기본
6. p04 — 실행시간 측정 타이머
7. 두 주제의 연결 — `@tool`도 데코레이터

---

# A. Tool Calling

LLM은 글을 잘 생성하지만 **큰 수 덧셈 같은 정확한 계산엔 약하다.** Tool Calling은 그런 일을 **미리 만든 함수(도구)에 맡기는** 방식이다. 핵심은 2단계로 나뉜다: **① LLM은 "어떤 도구를 어떤 인자로 부를지" 결정만** 하고, **② 실제 실행은 코드가** 한다. (LLM이 함수를 직접 돌리는 게 아니다.)

## 1. f31 — `@tool` 정의 · `bind_tools` · `tool_calls`

```python
from langchain_core.tools import tool

#1. 도구 정의
@tool
def add(aaa:int, bbb:int) -> int:
    """두 정수를 더합니다."""        # ★ 이 설명(docstring)을 LLM이 읽는다
    return aaa + bbb

print(add.name)          # add
print(add.description)   # 두 정수를 더합니다.
print(add.args)          # {'aaa': {...'type':'integer'}, 'bbb': {...}}
```

- **`@tool`**: 평범한 함수를 **LangChain 도구**로 바꾸는 데코레이터. 이때:
  - 함수 이름 → 도구 이름(`add.name`)
  - **독스트링(`"""..."""`) → 도구 설명(`add.description`)** — LLM은 이 설명을 보고 **언제 이 도구를 쓸지** 판단하므로, 독스트링을 잘 적는 게 중요하다.
  - **타입 힌트(`aaa:int`) → 인자 스키마(`add.args`)** — LLM이 넘길 값의 형식을 정의한다.

```python
#3. 도구 등록
llm = get_llm(max_tokens=200)
llm_with_tools = llm.bind_tools([add])   # LLM에 도구 목록을 묶는다

#4. 질문
response = llm_with_tools.invoke("233123과 1232451을 add로 더해줘.")

#5. 결과 확인
print("일반 응답 : ", response.content)        # 보통 비어 있음(도구를 쓰기로 했으니)
print("Tool Calls : ", response.tool_calls)    # ★ [{'name':'add','args':{'aaa':233123,'bbb':1232451},'id':...}]
```

- **`bind_tools([add])`**: LLM에 쓸 수 있는 도구 목록을 등록한다. 반환된 `llm_with_tools`는 도구를 호출할 수 있는 LLM이다.
- **`response.tool_calls`**: LLM이 "이 도구를(`name`) 이 인자로(`args`) 불러라"고 **결정한 내용**이다. 이 시점엔 **아직 계산이 실행된 게 아니다** — LLM은 호출 계획만 세웠다. 도구를 쓰기로 정하면 보통 `response.content`(일반 답변)는 비어 있다.

## 2. f32 — tool_calls 뜯어보기

```python
for call in response.tool_calls:
    print("tool 이름 : ", call['name'])   # add
    print("매개변수 : ", call['args'])    # {'aaa': 233123, 'bbb': 1232451}
    print("호출ID : ", call['id'])        # 호출 식별자
```
- `tool_calls`는 **리스트**(한 번에 여러 도구를 부를 수도 있어서)이고, 각 항목은 **`name`(도구 이름)·`args`(넘길 인자)·`id`(호출 식별자)** 를 담은 딕셔너리다. LLM이 뽑아준 이 정보로 다음 단계(실제 실행)를 한다.

## 3. f33 — 도구 실제 실행

```python
for call in response.tool_calls:
    result = add.invoke(call['args'])    # ★ LLM이 고른 인자로 함수를 직접 실행
    print("계산 결과 : ", result)        # 53
```
- **`add.invoke(call['args'])`**: LLM이 결정한 인자(`{'aaa':21,'bbb':32}`)를 도구에 넣어 **실제로 실행**한다. 이 줄에서 비로소 덧셈이 수행된다(2단계의 ②). 즉 **"LLM이 고른다(tool_calls) → 코드가 실행한다(invoke)"** 가 Tool Calling의 뼈대다.

## 4. f34 — 여러 도구와 tool_map

도구가 여러 개일 때, LLM이 **상황에 맞는 도구를 골라** 준다.

```python
@tool
def add(a, b): """두 정수를 더합니다""" ; return a + b
@tool
def multiply(a, b): """두 정수를 곱합니다""" ; return a * b
@tool
def subtract(a, b): """두 정수를 뺍니다""" ; return a - b

tools = [add, multiply, subtract]
tool_map = {tool.name : tool for tool in tools}   # ★ 이름 → 도구 딕셔너리

llm_with_tools = get_llm(max_tokens=300).bind_tools(tools)

for question in ["15와 27을 더해줘", "12와 8을 곱해줘", "200에서 150을 빼줘"]:
    response = llm_with_tools.invoke(question)
    for call in response.tool_calls:
        print("선택한 도구 : ", call['name'])          # 질문에 맞게 add/multiply/subtract
        result = tool_map[call['name']].invoke(call['args'])   # 이름으로 도구 찾아 실행
        print("실행 결과 : ", result)
```

- **`tool_map = {tool.name : tool for tool in tools}`**: **도구 이름 → 도구 객체** 딕셔너리(딕셔너리 컴프리헨션). LLM이 돌려준 `call['name']`(문자열)으로 **실제 도구 객체를 찾아** 실행하기 위한 매핑이다.
- 질문마다 LLM이 **알맞은 도구를 선택**한다("더해줘"→add, "곱해줘"→multiply). 도구를 하나하나 `if`로 분기하지 않고 `tool_map[call['name']]`으로 깔끔하게 실행한다.
- 이것이 AI 에이전트의 기초다: **여러 도구를 쥐여주면, LLM이 상황에 맞는 것을 골라 쓰고, 코드가 실행해 결과를 돌려준다.**

---

# B. 파이썬 데코레이터

`@tool`의 정체가 바로 **데코레이터**다. 데코레이터는 **기존 함수를 수정하지 않고, 그 함수에 새 기능을 덧붙이는** 문법이다.

## 5. p03 — 데코레이터 기본

```python
def deco(func):                 # 함수를 '받는' 함수
    def wrapper():
        print("함수 실행 전")    # 원래 함수 앞에 끼울 기능
        func()                  # 원래 함수 실행
        print("함수 실행 후")    # 원래 함수 뒤에 끼울 기능
    return wrapper              # 감싼 함수를 돌려줌

@deco
def hello2():
    print("안녕하세요.안녕하세요")

hello2()
# 함수 실행 전 / 안녕하세요.안녕하세요 / 함수 실행 후
```

- **`@deco`** 를 함수 위에 붙이면 `hello2 = deco(hello2)`와 같아진다. 즉 `hello2`가 **`wrapper`로 교체**되어, 호출하면 원래 코드 **앞뒤로 기능이 추가**된 채 실행된다.
- **구조 3단계**: ① `deco(func)`가 함수를 받고 → ② 안에서 `wrapper`를 정의해 `func()` 앞뒤에 코드를 넣고 → ③ `wrapper`를 반환. 원래 함수(`hello2`)의 코드는 **건드리지 않는다.**

## 6. p04 — 실행시간 측정 타이머

데코레이터는 **여러 함수에 공통 기능**(실행시간 측정·로그·권한 검사·예외 처리·호출 횟수 등)을 붙일 때 유용하다.

```python
import time
def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린시간 : ", round(end-start, 6), "초")
    return wrapper

@timer
def hello():
    print("안녕하세요"); time.sleep(1)

@timer
def study():
    print("NPU 공부중"); time.sleep(1.5)

hello()   # 안녕하세요 / 걸린시간 : 1.00xxxx 초
study()   # NPU 공부중 / 걸린시간 : 1.50xxxx 초
```

- **`@timer`** 하나를 여러 함수에 붙이면, 각 함수 실행을 자동으로 시간 측정해 준다. 함수마다 측정 코드를 중복해 넣을 필요가 없다 — **공통 기능을 한 곳에 모아 재사용**하는 게 데코레이터의 핵심 가치다.

## 7. 두 주제의 연결 — `@tool`도 데코레이터

- 오늘 A의 **`@tool`** 과 B의 **`@deco`·`@timer`** 는 같은 문법이다. `@tool`은 "평범한 함수를 **LangChain 도구로 감싸(이름·설명·인자 스키마를 뽑아) 바꿔주는**" 데코레이터다.
- 즉 `@tool def add(...)`는 `add = tool(add)`와 같고, 이 과정에서 함수가 `.name`·`.description`·`.args`·`.invoke()`를 갖춘 **도구 객체로 변신**한다. B(데코레이터 원리)를 알면 A(`@tool`)가 왜 그렇게 동작하는지가 이어진다.

---

## 한 줄 요약

> **Tool Calling**은 LLM이 못 하는 정확한 작업을 함수에 맡기는 방식으로, **`@tool`로 함수를 도구화**(독스트링→설명, 타입힌트→인자 스키마)하고 **`llm.bind_tools([...])`** 로 등록한다. 호출하면 LLM은 **`response.tool_calls`(name·args·id)로 "무엇을 어떻게 부를지 결정만"** 하고, 실제 실행은 **`도구.invoke(call['args'])`** 로 코드가 한다(2단계). 여러 도구는 **`tool_map={name:tool}`** 으로 이름→도구를 매핑해 LLM이 고른 것을 실행하며, 이것이 AI 에이전트의 기초다. **데코레이터**는 기존 함수를 수정하지 않고 기능을 덧붙이는 문법으로, `deco(func)`가 `wrapper`에서 `func()` 앞뒤에 코드를 넣고 반환하는 구조다(`@deco` = `f=deco(f)`). 실행시간 측정·로그 같은 **공통 기능 재사용**에 쓰며, 오늘 배운 **`@tool`도 바로 이 데코레이터**다.
