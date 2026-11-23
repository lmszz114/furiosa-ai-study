# 261123 복습 — LangChain 에이전트(create_agent)

> 지난 시간 Tool Calling에서는 "LLM이 도구를 고르면 → 코드가 직접 실행"을 **손으로** 처리했다. 오늘은 **`create_agent`** 로 그 과정을 **자동화**한다. 에이전트는 "생각 → 도구 호출 → 결과 확인 → 다시 생각"을 **스스로 반복**해, 여러 단계가 필요한 계산도 알아서 끝까지 수행한다.

## 목차
0. 에이전트란 — 지난 Tool Calling과의 차이
1. f35 — `create_agent` 기본 · 메시지 흐름 보기
2. f36 — 여러 도구 · 질문마다 알맞은 도구 선택
3. f37 — 다단계 연쇄 계산 · `recursion_limit`
4. f38 — 도구 실행 오류 처리
5. f39 — 완성형 대화 에이전트(함수화 + 입력 루프)

---

## 0. 에이전트란 — 지난 Tool Calling과의 차이

지난주 Tool Calling은 ① LLM이 `tool_calls`로 "무엇을 부를지" 정하면 → ② 내가 `도구.invoke()`로 실행하고 → ③ 결과를 다시 넘기는 과정을 **코드로 일일이** 짰다. **에이전트(Agent)는 이 "결정 → 실행 → 재판단" 루프를 스스로 돈다.** 도구를 쥐여주고 목표만 주면, 필요한 도구를 **순서대로 여러 번** 호출해 최종 답에 도달한다.

```python
from langchain.agents import create_agent   # pip install langchain

agent = create_agent(
    model = llm,                              # 사용할 LLM
    tools = [add, multiply],                  # 쥐여줄 도구 목록
    system_prompt="계산이 필요하면 도구를 사용하세요.",  # 역할·행동 지침
)
result = agent.invoke({"messages": [{"role":"user", "content":"33과 7을 곱한 후 4를 더해줘"}]})
```

- **`create_agent(model, tools, system_prompt)`**: LLM·도구·지침을 묶어 **에이전트**를 만든다. 도구는 지난 시간처럼 `@tool`로 정의한 함수들이다.
- **`system_prompt`**: 에이전트에게 주는 **역할·행동 지침**("계산이 필요하면 도구를 써라"). 에이전트가 어떻게 행동할지를 정한다.
- 내부적으로는 LangGraph의 그래프(LLM ↔ 도구 순환)가 자동으로 만들어져 돈다 — 8주차에 손으로 만들던 ToolNode·조건부 엣지 구조를 고수준 함수 하나로 대체한 것이다.

## 1. f35 — `create_agent` 기본 · 메시지 흐름 보기

"33과 7을 곱한 후 4를 더해줘"는 **두 단계**(곱하기 → 더하기) 계산이다. 에이전트가 이를 어떻게 처리하는지 `result["messages"]`에 쌓인 **대화 전체 흐름**으로 확인한다.

```python
from langchain_core.messages import AIMessage

result = agent.invoke({"messages": [{"role":"user", "content":"33과 7을 곱한 후 4를 더해줘"}]})

for i, msg in enumerate(result["messages"]):
    print(f"{i+1}번째 메시지")
    print("타입: ", type(msg).__name__)     # 메시지 종류
    print("내용: ", msg.content)
    if isinstance(msg, AIMessage) and msg.tool_calls:   # AI가 도구를 호출한 메시지라면
        for call in msg.tool_calls:
            print("도구 이름: ", call['name'])
            print("전달 인자: ", call['args'])
```

- **`result["messages"]`**: 에이전트가 거친 **모든 단계**가 메시지 리스트로 쌓인다. 메시지 종류(타입)는:
  - **HumanMessage**: 사용자 질문
  - **AIMessage**: LLM의 응답 — 도구를 부르기로 하면 여기에 `tool_calls`가 담긴다
  - **ToolMessage**: 도구를 실제 실행한 결과
  - 마지막 **AIMessage**: 모든 도구 결과를 종합한 최종 답변
- **`isinstance(msg, AIMessage) and msg.tool_calls`**: 그 메시지가 **"도구 호출을 담은 AI 메시지"** 인지 확인한다(사용자 메시지·도구 결과 메시지엔 `tool_calls`가 없으므로). 참이면 어떤 도구를 어떤 인자로 불렀는지 꺼낸다.
- 이 질문에서 에이전트는 `multiply(33,7)=231` → `add(231,4)=235`를 **스스로 순서대로** 호출한다. 최종 답은 `result["messages"][-1].content`(리스트의 마지막 메시지)다.

## 2. f36 — 여러 도구 · 질문마다 알맞은 도구 선택

도구를 넷(add·subtract·multiply·divide)으로 늘리고, 여러 질문을 돌려 **에이전트가 질문마다 맞는 도구를 고르는지** 본다.

```python
tools = [add, multiply, subtract, divide]
agent = create_agent(model=llm, tools=tools, system_prompt="계산이 필요하면 도구를 사용하세요.")

questions = ["100에서 35를 빼줘", "12와 8을 곱해줘", "100을 4로 나눠줘", "15와 27을 더해줘"]
for question in questions:
    result = agent.invoke({"messages": [{"role":"user", "content":question}]})
    print("최종 답변: ", result["messages"][-1].content)
```

- 각 `@tool`의 **독스트링**("첫번째 정수에서 두번째 정수를 뺍니다" 등)을 보고, 에이전트가 **"빼줘"→subtract, "곱해줘"→multiply** 식으로 알맞은 도구를 고른다. 도구가 많아도 사람이 분기하지 않고 에이전트가 판단한다.

## 3. f37 — 다단계 연쇄 계산 · `recursion_limit`

"65에서 35를 뺀 후, 그 결과에 3을 곱하고, 마지막으로 10 더해줘"처럼 **여러 단계가 연쇄**된 질문. 에이전트가 앞 결과를 다음 계산에 넘기며 **도구를 여러 번 이어서** 호출한다.

```python
questions = ("65에서 35를 뺀 후,"
             "그 결과에 3을 곱하고,"
             "마지막으로 10일 더해줘.")

result = agent.invoke(
    {"messages": [{"role":"user", "content":questions}]},
    config={"recursion_limit":20},      # ★ 에이전트 반복 횟수 상한
)
```

- **연쇄 처리**: `subtract(65,35)=30` → `multiply(30,3)=90` → `add(90,10)=100`을 **차례로** 호출한다. 각 단계 결과가 다음 도구의 입력이 된다(이것이 에이전트의 핵심 — 스스로 여러 스텝을 잇는다).
- **`config={"recursion_limit":20}`**: 에이전트가 "생각 ↔ 도구 호출"을 **최대 몇 번까지 반복할지** 정하는 상한이다. 복잡한 질문에서 단계가 많아지거나, 잘못돼 **무한 반복**에 빠지는 것을 막는 안전장치다. 단계 수에 비해 너무 작으면 중간에 멈춰 답을 못 낸다.
- **`(...문자열...)` 괄호 묶기**: 줄바꿈으로 나눠 쓴 문자열들은 파이썬에서 **자동으로 하나로 이어진다**(암시적 문자열 연결). 긴 질문·프롬프트를 여러 줄로 보기 좋게 적는 방법이다.

## 4. f38 — 도구 실행 오류 처리

"10을 0으로 나눠줘"는 **0으로 나누기** 라 `divide` 실행에서 오류가 난다. 오류에 대비하는 두 가지 장치를 쓴다.

```python
agent = create_agent(
    model=llm, tools=tools,
    system_prompt=("계산이 필요하면 도구를 사용하세요."
                   "도구 실행에 실패하면 오류를 설명하세요."),   # ① 프롬프트로 지침
)

try:                                        # ② 코드로도 예외 처리
    result = agent.invoke({"messages":[{"role":"user","content":"10을 0으로 나눠줘"}]},
                          config={"recursion_limit":20})
    print("최종 답변: ", result["messages"][-1].content)
except ValueError as e:
    print("계산오류: ", e)
except Exception as e:
    print("Agent 실행 오류: ", type(e).__name__, e)
```

- **두 겹의 대비**: ① **`system_prompt`** 에 "실패하면 오류를 설명하라"고 지침을 줘 에이전트가 사용자에게 상황을 말로 설명하게 하고, ② 그래도 예외가 터질 경우를 대비해 **`try/except`** 로 프로그램이 멈추지 않게 한다.
- **`except`를 둘로**: `ValueError`(값 관련 오류)를 먼저 잡고, 그 외 모든 예외는 `Exception`으로 받아 **오류 종류(`type(e).__name__`)와 내용**을 출력한다. 구체적 예외를 먼저, 포괄적 예외를 뒤에 두는 것이 순서다.

## 5. f39 — 완성형 대화 에이전트(함수화 + 입력 루프)

앞의 요소들을 합쳐 **실제 챗봇처럼 계속 대화하는** 형태로 완성한다. ① 역할을 구체화한 `system_prompt`, ② 호출을 `run_agent()` 함수로 정리, ③ `while`로 사용자 입력을 반복해서 받는다.

```python
agent = create_agent(
    model=llm, tools=tools,
    system_prompt=("넌 계산 전문 에이전트야."
                   "계산이 필요하면 등록된 도구를 사용해."
                   "여러 단계의 계산은 순서대로 수행해."
                   "계산 오류가 발생하면 사용자에게 설명해."),
)

def run_agent(question):                    # ② 호출 과정을 함수로
    result = agent.invoke({"messages":[{"role":"user","content":question}]},
                          config={"recursion_limit":20})
    # ... tool_calls 출력 ...
    print("최종 답변: ", result["messages"][-1].content)
    return result

while True:                                 # ③ 대화형 입력 루프
    question = input("\n계산할 질문을 입력하세요 (종료:q):")
    if question.lower() == 'q':             # q 입력 시 종료
        print("Agent 종료"); break
    if not question.strip():                # 빈 입력은 건너뜀
        continue
    try:
        run_agent(question)
    except Exception as e:
        print("Agent 실행 오류: ", type(e).__name__, e)
```

- **`system_prompt` 구체화**: "넌 계산 전문 에이전트야 / 여러 단계는 순서대로 / 오류는 설명해"처럼 **역할과 행동 규칙을 명확히** 줄수록 에이전트가 일관되게 동작한다.
- **`while True` + `input()`**: 사용자가 `q`를 칠 때까지 **계속 질문을 받아** 처리하는 대화 루프. **`question.strip()`** 이 비어 있으면(공백만 입력) `continue`로 건너뛰고, 매 질문을 `try/except`로 감싸 오류가 나도 프로그램이 죽지 않고 다음 입력을 받는다.
- 이것이 도구를 갖춘 **대화형 AI 에이전트의 기본 뼈대**다 — 역할 부여(프롬프트) + 도구 + 반복 실행 루프 + 오류 처리.

---

## 한 줄 요약

> **`create_agent(model, tools, system_prompt)`** 는 지난주 손으로 하던 Tool Calling("결정→실행→재판단")을 **자동 반복 루프**로 묶은 에이전트를 만든다. `agent.invoke({"messages":[...]})`의 결과 **`result["messages"]`** 에는 대화 전 과정(HumanMessage·도구를 부른 AIMessage·실행 결과 ToolMessage·최종 AIMessage)이 쌓이고, **`isinstance(msg, AIMessage) and msg.tool_calls`** 로 어떤 도구를 어떤 인자로 불렀는지 추적한다. 도구가 여럿이면 독스트링을 보고 **질문마다 맞는 도구를 스스로 고르며**(f36), 다단계 질문은 앞 결과를 다음 계산에 넘겨 **도구를 순서대로 여러 번** 호출한다(f37, `recursion_limit`으로 반복 상한). 오류는 **system_prompt 지침 + `try/except`** 두 겹으로 대비하고(f38), 역할을 구체화한 프롬프트·`run_agent()` 함수화·`while input()` 루프를 합치면 **대화형 계산 에이전트**가 된다(f39).
